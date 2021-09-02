from torch._C import device
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
        self.vectorizer = self._load_vectorizer(vectorizer_model_path)
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

        return embedding

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

    def word2vec_embedding(self, question_loader) :
        