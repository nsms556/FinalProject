from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext import Vectors

from khaiii import KhaiiiApi

from Utils.static import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Recommender(nn.Module) :
    def __init__(self) :
        super(Recommender, self).__init__()

        self.autoencoder = self._load_autoencoder(autoencoder_model_path)
        self.vectorizer, self.word_dict = self._load_vectorizer(vectorizer_model_path)
        self.tokenizer = KhaiiiApi()
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
            if genre : 
                auto_emb_gnr = self.pre_auto_emb_gnr.copy()
                for _id, _data, _dnr, _dtl_dnr in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    output = torch.cat([output.cpu(), _dnr, _dtl_dnr], dim=1)

                    output_df = pd.DataFrame(data=output.tolist(), index=_id.tolist())
                    auto_emb_gnr = pd.concat([auto_emb_gnr, output_df])

                    return auto_emb_gnr
            else :
                auto_emb = self.pre_auto_emb.copy()
                for _id, _data in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    
                    output_df = pd.DataFrame(data=output.tolist(), index=_id.tolist())
                    auto_emb = pd.concat([auto_emb, output_df])

                    return auto_emb

    def word2vec_embedding(self, question_data:pd.DataFrame) :
        def find_word_embed(words) :
            ret = []
            try :
                for word in words :
                    ret.append(self.word_dict[word])
            except KeyError :
                pass
  
            return ret

        w2v_emb = self.pre_w2v_emb.copy()

        p_ids = question_data['id']
        p_titles = question_data['plylst_title']
        p_tags = question_data['tags']
        p_dates = question_data['updt_date'].str[:7].str.split('-')

        question_data['tokens'] = p_titles + p_tags + p_dates
        question_data['emb_input'] = question_data['tokens'].map(lambda x : find_word_embed(x))

        outputs = []
        for e in question_data['emb_input'] :
            word_output = self.vectorizer(torch.LongTensor(e))
            if len(word_output) :
                output = torch.mean(word_output, axis=0)
            else :
                output = torch.zeros(200)
        outputs.append(output)
        outputs = torch.stack(outputs)

        w2v_emb = pd.concat([w2v_emb, pd.DataFrame(outputs.tolist(), index=p_ids)])

        return w2v_emb

        '''
            p_words = []
            for q in tqdm(question_data) :
                p_id = q['id']
                p_title = q['plylst_title']
                p_tags = q['tags']
                p_date = q['updt_date'][:7].split('-')

                p_title_tokens = self.tokenizer.analyze([p_title])
                if len(p_title_tokens):
                    p_title_tokens = p_title_tokens[0]
                else:
                    p_title_tokens = []

                p_words = p_title_tokens + p_tags + p_date

                word_input = np.array([])
                for p_word in p_words :
                    try :
                        x_word = np.append(word_input, self.word_dict[p_word])
                    except :
                        pass
                word_input = torch.from_numpy(x_word)

                word_output = self.vectorizer(word_input)

                if len(word_output) :
                    output = torch.mean(word_output, axis=0)
                else :
                    output = torch.zeros(200)
        '''