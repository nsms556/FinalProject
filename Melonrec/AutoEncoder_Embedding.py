import os
import sys
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Models.dataset import SongTagDataset, SongTagGenreDataset
from Models.autoencoder import AutoEncoder

from Utils.file import load_json, remove_file
from Utils.preprocessing import tags_encoding, song_filter_by_freq
from Utils.evaluate import mid_check
from Utils.static import *

class AutoEncoderHandler :
    def __init__(self, model_path:str = None) -> None:
        self.is_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.is_cuda else 'cpu'
        self.model = None
        
        if model_path is not None :
            self.load_model(model_path)

    def create_autoencoder(self, D_in, D_out, args) :
        self.model = AutoEncoder(D_in, args.dimension, D_out, dropout=args.dropout).to(self.device)

    def load_model(self, model_path) :
        self.model = torch.load(model_path)
        
    def save_model(self, model_path) :
        torch.save(self.model, model_path)

    def train_autoencoder(self, train_dataset, id2song_file_path, id2tag_file_path, question_dataset, answer_file_path, args) :
        id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
        id2song_dict = dict(np.load(id2song_file_path, allow_pickle=True).item())

        num_songs = train_dataset.num_songs
        num_tags = train_dataset.num_tags
        print(num_songs, num_tags)
        D_in = D_out = num_songs + num_tags

        q_dataloader = None
        check_every = 5

        if question_dataset is not None :
            q_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

        if self.model is None :
            self.create_autoencoder(D_in, D_out, args)

        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        remove_file(temp_fn)

        for epoch in range(args.epochs) :
            print('epoch : {}'.format(epoch))

            running_loss = 0.0
            for idx, (_id, _data) in tqdm(enumerate(dataloader), desc='training...') :
                _data = _data.to(self.device)

                optimizer.zero_grad()
                output = self.model(_data)

                loss = loss_func(output, _data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('loss: {:.4f} \n'.format(running_loss))

            if epoch % check_every == 0 and question_dataset is not None :
                mid_check(q_dataloader, self.model, tmp_results_path, answer_file_path, id2song_dict, id2tag_dict, self.is_cuda, num_songs)

    def autoencoder_plylsts_embeddings(self, playlist_data, genre=False, train=True):
        if genre:
            playlist_dataset = SongTagGenreDataset(playlist_data, tag2id_file_path, song2id_file_path)
        else:
            playlist_dataset = SongTagDataset(playlist_data, tag2id_file_path, song2id_file_path)

        plylst_embed_weight = []
        plylst_embed_bias = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name == 'encoder.1.weight':
                    plylst_embed_weight = param.data
                elif name == 'encoder.1.bias':
                    plylst_embed_bias = param.data

        playlist_loader = DataLoader(playlist_dataset, shuffle=True, batch_size=256, num_workers=4)

        if genre:
            if train :
                plylst_emb_with_bias = dict()
            else :
                plylst_emb_with_bias = dict(np.load(plylst_emb_gnr_path, allow_pickle=True).item())
            
            for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(playlist_loader, desc='')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                    output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]
        else:
            if train :
                plylst_emb_with_bias = dict()
            else :
                plylst_emb_with_bias = dict(np.load(plylst_emb_path, allow_pickle=True).item())
            for idx, (_id, _data) in enumerate(tqdm(playlist_loader, desc='')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        return plylst_emb_with_bias

'''
    def get_file_paths(args) :
        answer_file_path = None
        val_file_path = None
        test_file_path = None

        if args.mode == 0: 
            default_file_path = 'arena_data'
            model_postfix = 'local_val'
            train_file_path = f'{default_file_path}/orig/train.json'
            question_file_path = f'{default_file_path}/questions/val.json'
            answer_file_path = f'{default_file_path}/answers/val.json'
            
        elif args.mode == 1:
            default_file_path = 'res'
            model_postfix = 'val'
            train_file_path = f'{default_file_path}/train.json'
            val_file_path = f'{default_file_path}/val.json'
            
        elif args.mode == 2:
            default_file_path = 'res'
            model_postfix = 'test'
            train_file_path = f'{default_file_path}/train.json'
            val_file_path = f'{default_file_path}/val.json'
            test_file_path = f'{default_file_path}/test.json'
            
        else:
            print('mode error! local_val: 0, val: 1, test: 2')
            sys.exit(1)

        tag2id_file_path = f'{default_file_path}/tag2id_{model_postfix}.npy'
        id2tag_file_path = f'{default_file_path}/id2tag_{model_postfix}.npy'
        song2id_file_path = f'{default_file_path}/freq_song2id_thr{args.freq_thr}_{model_postfix}.npy'
        id2song_file_path = f'{default_file_path}/id2freq_song_thr{args.freq_thr}_{model_postfix}.npy'

        autoencoder_model_path = 'model/autoencoder_{}_{}_{}_{}_{}_{}.pkl'. \
            format(args.dimension, args.batch_size, args.learning_rate, args.dropout, args.freq_thr, model_postfix)

        return train_file_path, val_file_path, test_file_path, question_file_path, answer_file_path, \
        tag2id_file_path, id2tag_file_path, song2id_file_path, id2song_file_path, autoencoder_model_path
'''

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=41)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=4)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)

    args = parser.parse_args()
    print(args)
    mode = args.mode
    
    handler = AutoEncoderHandler()

    train_data = load_json(train_file_path)
    question_data = load_json(question_file_path)

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_encoding(train_data, tag2id_file_path, id2tag_file_path)

    if not (os.path.exists(song2id_file_path) & os.path.exists(id2song_file_path)):
        song_filter_by_freq(train_data, args.freq_thr, song2id_file_path, id2song_file_path)

    train_dataset = SongTagDataset(train_data, tag2id_file_path, song2id_file_path)

    handler.train_autoencoder(train_dataset, autoencoder_model_path, id2song_file_path, id2tag_file_path, question_dataset, answer_file_path)

    plylst_emb = handler.autoencoder_plylsts_embeddings(train_data, False, True)
    plylst_emb_gnr = handler.autoencoder_plylsts_embeddings(train_data, True, True)

    np.save(plylst_emb_path, plylst_emb)
    np.save(plylst_emb_gnr_path, plylst_emb_gnr)

    print('AutoEncoder Embedding Complete')
    