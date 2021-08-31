import os
import argparse

from AutoEncoder_Embedding import AutoEncoderHandler
from Word2vec_Embedding_Kakao import Word2VecHandler
import Calc_Similarity_Score as css

from Models.dataset import SongTagDataset

from Utils.static import *
from Utils.file import load_json
from Utils.preprocessing import song_filter_by_freq, tags_encoding

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=40)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=8)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-exist_tags_only', type=str, help="Y/N", default='Y')

    args = parser.parse_args()

    print('Load data')
    train_data = load_json(train_file_path)
    question_data = load_json(question_file_path)
    answer_data = load_json(answer_file_path)

    print('Create Embedding Handlers')
    autoencoder_handler = AutoEncoderHandler()
    word2vec_handler = Word2VecHandler()

    if not (os.path.exists(tag2id_file_path) and os.path.exists(id2song_file_path)) :
        tags_encoding(train_data, tag2id_file_path, id2tag_file_path)

    if not (os.path.exists(song2id_file_path) & os.path.exists(id2song_file_path)) :
        song_filter_by_freq(train_data, args.freq_thr, song2id_file_path, id2song_file_path)

    train_dataset = SongTagDataset(train_data, tag2id_file_path, song2id_file_path)
    question_dataset = SongTagDataset(question_data, tag2id_file_path, song2id_file_path)

    autoencoder_handler.train_autoencoder(train_dataset, id2song_file_path, id2tag_file_path, question_dataset, answer_file_path, args)

    autoencoder_path = autoencoder_model_path.format(args.dimension, args.batch_size, args.learning_rate, args.dropout, args.freq_thr)
    autoencoder_handler.save_model(autoencoder_path)

    word2vec_handler.train_vectorizer(train_file_path, genre_meta_file_path, True)
    word2vec_handler.vectorizer.save_model(vectorizer_model_path)