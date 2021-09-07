from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Models.dataset import SongTagDataset, SongTagGenreDataset

from Utils.file import remove_file
from Utils.evaluate import mid_check
from Utils.static import *


class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        encoder_layer = nn.Linear(D_in, H, bias=True)
        decoder_layer = nn.Linear(H, D_out, bias=True)

        nn.init.xavier_uniform_(encoder_layer.weight)
        nn.init.xavier_uniform_(decoder_layer.weight)

        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        encoder_layer,
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU())
        self.decoder = nn.Sequential(
                        decoder_layer,
                        nn.Sigmoid())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder


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

    def export_encoder_layer(self, weights_path) :
        torch.save(self.model.encoder[1], weights_path)

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
        playlist_data = pd.DataFrame(playlist_data)
        if genre:
            playlist_dataset = SongTagGenreDataset(playlist_data)
        else:
            playlist_dataset = SongTagDataset(playlist_data)

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