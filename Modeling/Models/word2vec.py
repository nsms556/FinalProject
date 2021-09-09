# Visualization library
from tqdm import tqdm

# Data library
import numpy as np
import pandas as pd

# NLP library
from khaiii import KhaiiiApi
from gensim.models import Word2Vec

# Utils
from Utils.file import load_json
from Utils.static import vectorizer_weights_path, plylst_w2v_emb_path


class Kakao_Tokenizer :
    def __init__(self) -> None:
        self.tokenizer = KhaiiiApi()
        self.using_pos = ['NNG','SL','NNP','MAG','SN']  # 일반 명사, 외국어, 고유 명사, 일반 부사, 숫자

    def re_sub(self, series: pd.Series) -> pd.Series:
        series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
        series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
        series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
        series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
        
        return series

    def flatten(self, list_of_list) :
        flatten = [j for i in list_of_list for j in i]
        
        return flatten

    def get_token(self, title: str) :
        if len(title)== 0 or title== ' ':  # 제목이 공백인 경우 tokenizer에러 발생
            return []

        result = self.tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in tqdm(result) for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
        
        return result

    def get_all_tags(self, df: pd.DataFrame) :
        tag_list = df['tags'].values.tolist()
        tag_list = self.flatten(tag_list)
        
        return tag_list

    def filter_by_exist_tag(self, tokens, exist_tags) :
        token_tag = [self.get_token(x) for x in exist_tags]
        token_itself = list(filter(lambda x: len(x)==1, token_tag))
        token_itself = self.flatten(token_itself)
        unique_tag = set(token_itself)
        unique_word = [x[0] for x in unique_tag]

        tokens = tokens.map(lambda x: list(filter(lambda x: x[0] in unique_word, x)))
        tokens = tokens.map(lambda x : list(set(x)))

        return tokens

    def sentences_to_tokens(self, sentences, exist_tags=None) :
        token_series = self.re_sub(pd.Series(sentences))
        token_series = token_series.map(lambda x: self.get_token(x))
        token_series = token_series.map(lambda x: list(filter(lambda x: x[1] in self.using_pos, x)))

        if exist_tags is not None :
            token_series = self.filter_by_exist_tag(token_series, exist_tags)

        tokenized_stc = token_series.map(lambda x: [tag[0] for tag in x]).tolist()
        
        return tokenized_stc

class Str2Vec :
    def __init__(self, train_data=None, size=200, window=5, min_count=2, workers=8, sg=1, hs=1):
        self.model = Word2Vec(size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)
        
        if train_data != None :
            self.build_vocab(train_data)

    def build_vocab(self, train_data) :
        self.model.build_vocab(train_data)

    def load_model(self, model_fn):
        self.model = Word2Vec.load(model_fn)

    def save_embeddings(self, emb_fn):
        word_vectors = self.model.wv

        vocabs = []
        vectors = []
        for key in word_vectors.vocab:
            vocabs.append(key)
            vectors.append(word_vectors[key])

        df = pd.DataFrame()
        df['voca'] = vocabs
        df['vector'] = vectors

        df.to_csv(emb_fn, index=False)

    def save_weights(self, md_fn):
        self.model.wv.save_word2vec_format(md_fn)

    def show_similar_words(self, word, topn):
        print(self.model.most_similar(positive=[word], topn=topn))

class Word2VecHandler :
    def __init__(self, model_path=None) :
        self.tokenizer = Kakao_Tokenizer()
        self.vectorizer = Str2Vec()

        if model_path != None :
            self.vectorizer.load_model(model_path)

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

    def train_vectorizer(self, train_data, genre_file_path, exist_tags_only=True):
        print('Make Sentences')
        sentences = self.make_input4tokenizer(train_data, genre_file_path)
        if not sentences:
            raise Exception('Sentences not found')
        
        print('Tokenizing')
        if exist_tags_only :    # kakao filtered #
            tokenized_sentences = self.tokenizer.sentences_to_tokens(sentences, self.tokenizer.get_all_tags(pd.DataFrame(train_data)))
        else :                  # kakao non-filtered #
            tokenized_sentences = self.tokenizer.sentences_to_tokens(sentences)

        print("Train vectorizer...")
        print("Save Path : {}".format(vectorizer_weights_path))
        self.vectorizer = Str2Vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)

    def get_plylsts_embeddings(self, playlist_data, exist=None, train=True):
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