import os
import sys
import argparse
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils import tags_ids_convert, save_freq_song_id_dict


class TitleTokenizer:
    def __init__(self):
        pass
    
    def make_input_file(self, input_fn, sentences):
        with open(input_fm, 'w', encoding='utf8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
    
    def train_tokenizer(self, input_fn, prefix, vocab_size, model_type):
        templates = '--input={} --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --model_prefix={} --vocab_size={} --character_coverage=1.0 --model_type={}'
        cmd = templates.format(
            input_fn,
            prefix,
            vocab_size,
            model_type
        )
        
        spm.SentencePieceTrainer.Train(cmd)
        print(f"tokenizer model {prefix}.model is trained")
    
    def get_tokens(self, sp, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokens = sp.EncodeAsPieces(sentence)
            new_tokens = []
            for token in tokens:
                token = token.replace("_", "")
                if len(token) > 1:
                    new_tokens.append(token)
            if len(new_tokens) > 1:
                tokenized_sentences.append(new_tokens)
        return tokenized_sentences


if __name__ == "__main__":
    # 0. 하이퍼 파라미터 입력
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=41)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=20)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-mode', type=int, help="local_val: 0, val: 1, test: 2", default=2)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    freq_thr = args.freq_thr
    mode = args.mode
    
    # 1. Embedding 준비
    
    # 1-1. tag2id & id2tag
    # Autoencoder의 input: song, tag binary vector의 concatenate, tags는 str이므로 id로 변형할 필요 있음
    tag2id_file_path = f'{default_file_path}/tag2id_{model_postfix}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{model_postfix}.npy'
    
    # 관련 데이터들이 없으면 default file path에 새로 만들음
    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_data, tag2id_file_path, id2tag_file_path)

    # 1-2. freq_song2id & id2freq_song
    # Song이 너무 많기 때문에 frequency에 기반하여 freq_thr번 이상 등장한 곡들만 남김, 남은 곡들에게 새로운 id 부여
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr{freq_thr}_{model_postfix}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr{freq_thr}_{model_postfix}.npy'

    # 관련 데이터들이 없으면 default file path에 새로 만들음
    if not (os.path.exists(prep_song2id_file_path) & os.path.exists(id2prep_song_file_path)):
        save_freq_song_id_dict(train_data, freq_thr, default_file_path, model_postfix)

    # 2. Embedding 실행
    
    # 2-0. dataset
    train_dataset = SongTagDataset(train_data, tag2id_file_path, prep_song2id_file_path)
    if question_data is not None:
        question_dataset = SongTagDataset(question_data, tag2id_file_path, prep_song2id_file_path)
    
    # 2-1. Song One-hot Vector
    # autoencoder model
    model_file_path = f'model/autoencoder_{H}_{batch_size}_{learning_rate}_{dropout}_{freq_thr}_{model_postfix}'

    train(train_dataset, model_file_path, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path)
    
    # 2-2. Tag One-hot Vector
    # word2vec
    vocab_size = 24000
    method = 'bpe'
    if model_postfix == 'val':
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
    elif model_postfix == 'test':
        default_file_path = 'res'
        val_file_path = 'res/val.json'
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
    elif model_postfix == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        default_file_path = f'{default_file_path}/orig'

    genre_file_path = 'res/genre_gn_all.json'

    tokenize_input_file_path = f'model/tokenizer_input_{method}_{vocab_size}_{model_postfix}.txt'

    if model_postfix == 'local_val':
        val_file_path = None
        test_file_path = None
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif model_postfix == 'val':
        test_file_path = None
        val_file_path = question_file_path
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif model_postfix == 'test':
        val_file_path = val_file_path
        test_file_path = question_file_path
        train = load_json(train_file_path)
        val = load_json(val_file_path)
        test = load_json(test_file_path)
        train = train + val
        question = test

    train_tokenizer_w2v(train_file_path, val_file_path, test_file_path, genre_file_path, tokenize_input_file_path,
                        model_postfix)