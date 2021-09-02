from torch._C import dtype
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext import Vectors

from khaiii import KhaiiiApi

from Models.word2vec import Kakao_Tokenizer

from Utils.static import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Recommender(nn.Module) :
    def __init__(self) :
        super(Recommender, self).__init__()

        self.autoencoder = self._load_autoencoder(autoencoder_model_path)
        self.vectorizer, self.word_dict = self._load_vectorizer('Weights/w2v')
        self.tokenizer = Kakao_Tokenizer()
        self.cos = nn.CosineSimilarity(dim=1)

        self.pre_auto_emb = pd.DataFrame(np.load(plylst_emb_path, allow_pickle=True).item()).T
        self.pre_auto_emb_gnr = pd.DataFrame(np.load(plylst_emb_gnr_path, allow_pickle=True).item()).T
        self.pre_w2v_emb = pd.DataFrame(np.load(plylst_w2v_emb_path, allow_pickle=True).item()).T

    def _load_autoencoder(self, model_path) :
        autoencoder = torch.load(model_path)

        return autoencoder

    def _load_vectorizer(self, model_path) :
        vectors = Vectors(name=model_path)
        embedding = nn.Embedding.from_pretrained(vectors.vectors, freeze=False)

        return embedding, pd.Series(vectors.stoi)

    def autoencoder_embedding(self, question_loader:DataLoader, genre:bool) :
        with torch.no_grad() :
            output_df = pd.DataFrame()
            if genre : 
                for _id, _data, _dnr, _dtl_dnr in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    output = torch.cat([output.cpu(), _dnr, _dtl_dnr], dim=1)

                    output_df = pd.concat([output_df, pd.DataFrame(data=output.tolist(), index=_id.tolist())])

                return output_df
            else :
                for _id, _data in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    
                    output_df = pd.concat([output_df, pd.DataFrame(data=output.tolist(), index=_id.tolist())])
                    
                return output_df

    def word2vec_embedding(self, question_data:pd.DataFrame) :
        def find_word_embed(words) :
            ret = []
            try :
                for word in words :
                    ret.append(self.word_dict[word])
            except KeyError :
                pass
  
            return ret

        p_ids = question_data['id']
        p_token = question_data['plylst_title'].map(lambda x : self.tokenizer.sentences_to_tokens(x)[0])
        p_tags = question_data['tags']
        p_dates = question_data['updt_date'].str[:7].str.split('-')
       
        question_data['tokens'] = p_token + p_tags + p_dates
        question_data['emb_input'] = question_data['tokens'].map(lambda x : find_word_embed(x))

        outputs = []
        for e in question_data['emb_input'] :
            _data = torch.LongTensor(e).to(device)
            word_output = self.vectorizer(_data)
            if len(word_output) :
                output = torch.mean(word_output, axis=0)
            else :
                output = torch.zeros(200)
            outputs.append(output)
        outputs = torch.stack(outputs)

        output_df = pd.DataFrame(outputs.tolist(), index=p_ids)

        return output_df

    def calc_similarity(self, question_df, train_df) :
        train_tensor = torch.from_numpy(train_df.values).to(device)
        question_tensor = torch.from_numpy(question_df.values).to(device)

        scores = torch.zeros([question_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64).to(device)
        for idx, vector in enumerate(question_tensor) :
            output = self.cos(vector.reshape(1, -1), train_tensor)
            scores[idx] = output

        scores = torch.sort(scores, descending=True)
        sorted_scores, sorted_idx = scores.values.cpu().numpy(), scores.indices.cpu().numpy()

        s = pd.DataFrame(sorted_scores, index=question_df.index)
        i = pd.DataFrame(sorted_idx, index=question_df.index).applymap(lambda x : train_df.index[x])

        return pd.DataFrame([pd.Series(list(zip(i.loc[idx], s.loc[idx]))) for idx in question_df.index])

    def calc_score(self) :
        