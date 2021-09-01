import torch
import torch.nn as nn
from torchtext import Vectors

from Utils.static import *

class Recommender(nn.Module) :
    def __init__(self) :
        super(Recommender, self).__init__()

        self.autoencoder = self._load_autoencoder(autoencoder_model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_model_path)
        self.similarity = nn.CosineSimilarity()

    def _load_autoencoder(self, model_path) :
        autoencoder = torch.load(model_path)

        return autoencoder

    def _load_vectorizer(self, model_path) :
        vectors = Vectors(name=model_path)
        embedding = nn.Embedding.from_pretrained(vectors.vectors, freeze=False)

        return embedding

    