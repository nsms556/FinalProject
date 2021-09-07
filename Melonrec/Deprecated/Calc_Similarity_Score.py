import argparse
from collections import defaultdict
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

from Utils.file import load_json
from Utils.static import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pcc(_x, _y):
    vx = _x - torch.mean(_x)
    vy = _y - torch.mean(_y, axis=1).reshape(-1, 1)
    return torch.sum((vx * vy), axis=1) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum((vy ** 2), axis=1)))

def euclidean(_x, _y):
    return torch.sqrt(torch.sum((_y - _x) ** 2, axis=1))

def calculate_score(train, question, embedding, score_type) :
    all_train_ids = [plylst['id'] for plylst in train]
    all_val_ids = [plylst['id'] for plylst in question]

    train_ids = []
    train_embs = []
    val_ids = []
    val_embs = []

    for plylst_id, emb in tqdm(embedding.items()):
        if plylst_id in all_train_ids:
            train_ids.append(plylst_id)
            train_embs.append(emb)
        elif plylst_id in all_val_ids:
            val_ids.append(plylst_id)
            val_embs.append(emb)

    cos = nn.CosineSimilarity(dim=1)
    train_tensor = torch.tensor(train_embs).to(device)
    val_tensor = torch.tensor(val_embs).to(device)

    scores = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64)
    sorted_idx = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.int32)

    for idx, val_vector in enumerate(tqdm(val_tensor)):
        if score_type == 'pcc':
            output = pcc(val_vector.reshape(1, -1), train_tensor)
        elif score_type == 'cos':
            output = cos(val_vector.reshape(1, -1), train_tensor)
        elif score_type == 'euclidean':
            output = euclidean(val_vector.reshape(1, -1), train_tensor)
        index_sorted = torch.argsort(output, descending=True)
        scores[idx] = output
        sorted_idx[idx] = index_sorted

    results = defaultdict(list)
    for i, val_id in enumerate(tqdm(val_ids)):
        for j, train_idx in enumerate(sorted_idx[i][:1000]):
            results[val_id].append((train_ids[train_idx], scores[i][train_idx].item()))

    return results

def save_autoencoder_score(train, question, autoencoder_emb, score_type, include_genre) :
    score = calculate_score(train, question, autoencoder_emb, score_type)

    if include_genre:
        np.save(autoencoder_gnr_score_file_path, score)
    else:
        np.save(autoencoder_score_file_path, score)

def save_word2vec_score(train, question, word2vec_emb, score_type) :
    score = calculate_score(train, question, word2vec_emb, score_type)

    np.save(word2vec_score_file_path, score)


if __name__ == '__main__' :
    train = load_json(train_file_path)
    question = load_json(question_file_path)

    autoencoder_emb = np.load(plylst_emb_path, allow_pickle=True).item()
    autoencoder_emb_gnr = np.load(plylst_emb_gnr_path, allow_pickle=True).item()
    word2vec_emb = np.load(plylst_w2v_emb_path, allow_pickle=True).item()

    save_autoencoder_score(train, question, autoencoder_emb, 'cos', False)
    save_autoencoder_score(train, question, autoencoder_emb_gnr, 'cos', True)
    save_word2vec_score(train, question, word2vec_emb, 'cos')

    print('Calculate Similarity Score Complete')