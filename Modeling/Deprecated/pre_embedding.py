import numpy as np

from AutoEncoder_Embedding import AutoEncoderHandler
from Word2vec_Embedding_Kakao import Word2VecHandler

from Utils.static import *
from Utils.file import load_json

if __name__ == '__main__' :
    train_data = load_json(train_file_path)
    
    auto_h = AutoEncoderHandler(autoencoder_model_path)

    plylst_emb = auto_h.autoencoder_plylsts_embeddings(train_data, False, True)
    plylst_emb_gnr = auto_h.autoencoder_plylsts_embeddings(train_data, True, True)

    word_h = Word2VecHandler(vectorizer_model_path)

    plylst_w2v_emb = word_h.get_plylsts_embeddings(train_data, train=True)
    
    np.save(plylst_emb_path, plylst_emb)
    np.save(plylst_emb_gnr_path, plylst_emb_gnr)
    np.save(plylst_w2v_emb_path, plylst_w2v_emb)
