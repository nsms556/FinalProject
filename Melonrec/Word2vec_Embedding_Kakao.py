import sys
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

from Utils.file import load_json
from Models.word2vec import Kakao_Tokenizer, Str2Vec
from Utils.static import *

class Word2VecHandler :
    def __init__(self) :
        self.tokenizer = Kakao_Tokenizer()
        self.vectorizer = None

    def make_input4tokenizer(self, train_file_path, genre_file_path):
        def _wv_genre(genre):
            genre_dict = dict()
            for code, value in genre:
                code_num = int(code[2:])
                if not code_num % 100:
                    cur_genre = value
                    genre_dict[cur_genre] = []
                else:
                    value = ' '.join(value.split('/'))
                    genre_dict[cur_genre].append(value)

            genre_sentences = []
            for key, sub_list in genre_dict.items():
                sub_list = genre_dict[key]
                key = ' '.join(key.split('/'))
                if not len(sub_list):
                   continue
                for sub in sub_list:
                    genre_sentences.append(key+' '+sub)

            return genre_sentences

        try:
            playlists = load_json(train_file_path)

            genre_all = load_json(genre_file_path)
            genre_all_lists = []
            for code, gnr in genre_all.items():
                if gnr != '세부장르전체':
                    genre_all_lists.append([code, gnr])
            genre_all_lists = np.asarray(genre_all_lists)
            genre_stc = _wv_genre(genre_all_lists)

            sentences = []
            for playlist in playlists:
                title_stc = playlist['plylst_title']
                tag_stc = ' '.join(playlist['tags'])
                date_stc = ' '.join(playlist['updt_date'][:7].split('-'))
                sentences.append(' '.join([title_stc, tag_stc, date_stc]))

            sentences = sentences + genre_stc
        except Exception as e:
            print(e.with_traceback())
            return False

        return sentences

    def train_vectorizer(self, train_file_path, genre_file_path, exist_tags_only=True):
        print('Make Sentences')
        sentences = self.make_input4tokenizer(train_file_path, genre_file_path)
        if not sentences:
            raise Exception('Sentences not found')
        
        print('Tokenizing')
        if exist_tags_only :    # kakao filtered #
            tokenized_sentences = self.tokenizer.sentences_to_tokens(sentences, self.tokenizer.get_all_tags(pd.read_json(train_file_path)))
        else :                  # kakao non-filtered #
            tokenized_sentences = self.tokenizer.sentences_to_tokens(sentences)

        vectorizer_name = 'model/vectorizer_{}.model'.format(self.model_postfix)
        print("start train_vectorizer.... name : {}".format(vectorizer_name))

        self.vectorizer = Str2Vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)

    def get_plylsts_embeddings(self,playlist_data, exist=None, train=True):
        print('saving embeddings')
        if train :
            t_plylst_title_tag_emb = {}  # plylst_id - vector dictionary
        else :
            if exist is not None :
                t_plylst_title_tag_emb = exist
            else :
                t_plylst_title_tag_emb = dict(np.load(plylst_w2v_emb_path, allow_pickle=True).item())

        for plylst in tqdm(playlist_data):
            p_id = plylst['id']

            p_title = plylst['plylst_title']
            p_title_tokens = self.tokenizer.sentences_to_tokens([p_title])
            if len(p_title_tokens):
                p_title_tokens = p_title_tokens[0]
            else:
                p_title_tokens = []

            p_tags = plylst['tags']
            p_times = plylst['updt_date'][:7].split('-')
            p_words = p_title_tokens + p_tags + p_times

            word_embs = []
            for p_word in p_words:
                try:
                    word_embs.append(self.vectorizer.model.wv[p_word])
                except KeyError:
                    pass

            if len(word_embs):
                p_emb = np.average(word_embs, axis=0).tolist()
            else:
                p_emb = np.zeros(200).tolist()

            t_plylst_title_tag_emb[p_id] = p_emb

        return t_plylst_title_tag_emb

'''
    def get_file_paths(model_postfix):
        genre_file_path = 'res/genre_gn_all.json'
        tokenize_input_file_path = f'model/tokenizer_input_{model_postfix}.txt'

        if model_postfix == 'val':
            default_file_path = 'res'
            train_file_path = 'res/train.json'
            question_file_path = 'res/val.json'
            val_file_path = question_file_path
            test_file_path = None

        elif model_postfix == 'test':
            default_file_path = 'res'
            train_file_path = 'res/train.json'
            question_file_path = 'res/test.json'
            val_file_path = 'res/val.json'
            test_file_path = question_file_path

        elif model_postfix == 'local_val':
            default_file_path = 'arena_data'
            train_file_path = f'{default_file_path}/orig/train.json'
            question_file_path = f'{default_file_path}/questions/val.json'
            val_file_path = None
            test_file_path = None

        return train_file_path, question_file_path, val_file_path, test_file_path, genre_file_path, tokenize_input_file_path
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exist_tags_only', type=str, help="Y/N", default='Y')
    
    args = parser.parse_args()
    print(args)

    default_file_path = 'Data'

    if args.exist_tags_only== 'N':
        exist_filter = False
    else :
        exist_filter = True
    
    handler = Word2VecHandler()
    handler.train_vectorizer(train_file_path, genre_meta_file_path, exist_filter)

    train = load_json(train_file_path)
    question = load_json(question_file_path)

    plylst_title_tag_emb = handler.get_plylsts_embeddings(train, train=True)
    plylst_title_tag_emb = handler.get_plylsts_embeddings(question, exist=plylst_title_tag_emb, train=False)

    np.save(plylst_w2v_emb_path, plylst_title_tag_emb)

    print('Word2Vec Embedding Complete')
