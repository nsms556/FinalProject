import pandas as pd

import torch.nn as nn

from khaiii import KhaiiiApi
from gensim.models import Word2Vec


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
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
        
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
    def __init__(self, train_data, size=200, window=5, min_count=2, workers=8, sg=1, hs=1):
        self.model = Word2Vec(size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)
        self.model.build_vocab(train_data)

    def set_model(self, model_fn):
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

    def save_model(self, md_fn):
        self.model.save(md_fn)
        print("word embedding model {} is trained".format(md_fn))

    def show_similar_words(self, word, topn):
        print(self.model.most_similar(positive=[word], topn=topn))