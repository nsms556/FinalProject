import datetime as dt
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors

from Models.word2vec import Kakao_Tokenizer
from Models.dataset import SongTagDataset, SongTagGenreDataset

from Utils.file import load_json, write_json
from Utils.static import *
from Utils.preprocessing import DicGenerator, most_popular, remove_seen, most_similar, most_similar_emb


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Recommender(nn.Module) :
    def __init__(self, auto_weights=autoencoder_model_path, w2v_weights=vectorizer_weights_path) :
        super(Recommender, self).__init__()

        self.autoencoder = self._load_autoencoder(auto_weights)
        self.vectorizer, self.word_dict = self._load_vectorizer(w2v_weights)
        self.tokenizer = Kakao_Tokenizer()
        self.cos = nn.CosineSimilarity(dim=1)

        self.pre_auto_emb = pd.DataFrame(np.load(plylst_emb_path, allow_pickle=True).item()).T
        self.pre_auto_emb_gnr = pd.DataFrame(np.load(plylst_emb_gnr_path, allow_pickle=True).item()).T
        self.pre_w2v_emb = pd.DataFrame(np.load(plylst_w2v_emb_path, allow_pickle=True).item()).T

        self._load_dictionary()

    def _load_autoencoder(self, model_path) :
        autoencoder = torch.load(model_path)

        return autoencoder

    def _load_vectorizer(self, model_path) :
        vectors = Vectors(name=model_path)
        embedding = nn.Embedding.from_pretrained(vectors.vectors, freeze=False).to(device)

        return embedding, pd.Series(vectors.stoi)

    def _load_dictionary(self) :
        train_data = load_json(train_file_path)
        song_meta = load_json(song_meta_file_path)
        
        self.song_plylst_dic, self.song_tag_dic, self.plylst_song_dic, self.plylst_tag_dic, self.tag_plylst_dic, self.tag_song_dic, _, self.song_artist_dic = DicGenerator(train_data, song_meta)
        self.freq_song = set(dict(np.load(song2id_file_path, allow_pickle=True).item()).keys())
        _, self.song_popular = most_popular(train_data, 'songs', 200)
        _, self.tag_popular = most_popular(train_data, 'tags', 20)

    def autoencoder_embedding(self, question_dataset, genre:bool) :
        question_loader = DataLoader(question_dataset, batch_size=256, num_workers=8)

        with torch.no_grad() :
            output_df = pd.DataFrame()
            if genre : 
                for _id, _data, _dnr, _dtl_dnr in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    output = torch.cat([output.cpu(), _dnr, _dtl_dnr], dim=1)

                    output_df = pd.concat([output_df, pd.DataFrame(data=output.numpy(), index=_id.tolist())])

                return output_df
            else :
                for _id, _data in tqdm(question_loader) :
                    _data = _data.to(device)
                    output = self.autoencoder.encoder[1](_data)
                    
                    output_df = pd.concat([output_df, pd.DataFrame(data=output.cpu().numpy(), index=_id.tolist())])
                    
                return output_df

    def word2vec_embedding(self, question_data:pd.DataFrame) :
        def find_word_embed(words) :
            ret = []
            for word in words :
                try :
                    ret.append(self.word_dict[word])
                except KeyError :
                    pass
  
            return ret

        p_ids = question_data['id']
        p_token = question_data['plylst_title'].map(lambda x : self.tokenizer.sentences_to_tokens(x)[0])
        p_tags = question_data['tags']
        p_dates = question_data['updt_date'].str[:7].str.split('-')
       
        question_data['tokens'] = p_token + p_tags + p_dates
        question_data['emb_input'] = question_data['tokens'].map(lambda x : find_word_embed(x))

        outputs = []
        for e in question_data['emb_input'] :
            _data = torch.LongTensor(e).to(device)
            word_output = self.vectorizer(_data)
            if len(word_output) :
                output = torch.mean(word_output, axis=0)
            else :
                output = torch.zeros(200).to(device)
            outputs.append(output)
        outputs = torch.stack(outputs)

        output_df = pd.DataFrame(outputs.tolist(), index=p_ids)

        return output_df

    def calc_similarity(self, question_df, train_df) :
        df_id = question_df.index
        train_tensor = torch.from_numpy(train_df.values).to(device)
        question_tensor = torch.from_numpy(question_df.values).to(device)

        scores = torch.zeros([question_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64).to(device)
        for idx, vector in enumerate(question_tensor) :
            output = self.cos(vector.reshape(1, -1), train_tensor)
            scores[idx] = output

        scores = torch.sort(scores, descending=True)
        sorted_scores, sorted_idx = scores.values.cpu().numpy(), scores.indices.cpu().numpy()

        s = pd.DataFrame(sorted_scores, index=question_df.index)
        i = pd.DataFrame(sorted_idx, index=question_df.index).applymap(lambda x : train_df.index[x])

        return pd.DataFrame([pd.Series(list(zip(i.loc[idx], s.loc[idx]))) for idx in df_id], index=df_id)
    
    def similarity_by_auto(self, question_df, genre:bool) :
        with torch.no_grad() :
            if genre :
                train_tensor = torch.from_numpy(self.pre_auto_emb_gnr.values).to(device)
                question_dataset = SongTagGenreDataset(question_df)
                question_loader = DataLoader(question_dataset, batch_size=256, num_workers=8)
                for _id, _data, _dnr, _dtl_dnr in tqdm(question_loader) :
                    _data = _data.to(device)
                    auto_emb = self.autoencoder.encoder[1](_data)
                    auto_emb = torch.cat([auto_emb, _dnr.to(device), _dtl_dnr.to(device)], dim=1)
            else :
                train_tensor = torch.from_numpy(self.pre_auto_emb.values).to(device)
                question_dataset = SongTagDataset(question_df)
                question_loader = DataLoader(question_dataset, batch_size=256, num_workers=8)
                for _id, _data in tqdm(question_loader) :
                    _data = _data.to(device)
                    auto_emb = self.autoencoder.encoder[1](_data)

        scores = torch.zeros([auto_emb.shape[0], train_tensor.shape[0]], dtype=torch.float64).to(device)
        for idx, vector in enumerate(auto_emb) :
            output = self.cos(vector.reshape(1, -1), train_tensor)
            scores[idx] = output

        scores = torch.sort(scores, descending=True)
        sorted_scores, sorted_idx = scores.values.cpu().numpy(), scores.indices.cpu().numpy()

        s = pd.DataFrame(sorted_scores, index=question_df['id'])
        if genre :
            i = pd.DataFrame(sorted_idx, index=question_df['id']).applymap(lambda x : self.pre_auto_emb_gnr.index[x])
        else :
            i = pd.DataFrame(sorted_idx, index=question_df['id']).applymap(lambda x : self.pre_auto_emb.index[x])

        return pd.DataFrame([pd.Series(list(zip(i.loc[idx], s.loc[idx]))) for idx in question_df['id']], index=question_df['id'])        
    
    def similarity_by_w2v(self, question_df) :
        def find_word_embed(words) :
            ret = []
            for word in words :
                try :
                    ret.append(self.word_dict[word])
                except KeyError :
                    pass
                
            return ret

        p_ids = question_df['id']
        p_token = question_df['plylst_title'].map(lambda x : self.tokenizer.sentences_to_tokens(x)[0])
        p_tags = question_df['tags']
        p_dates = question_df['updt_date'].str[:7].str.split('-')

        question_df['tokens'] = p_token + p_tags + p_dates
        question_df['emb_input'] = question_df['tokens'].map(lambda x : find_word_embed(x))

        outputs = []
        for e in question_df['emb_input'] :
            _data = torch.LongTensor(e).to(device)
            with torch.no_grad() :
                word_output = self.vectorizer(_data)
            if len(word_output) :
                output = torch.mean(word_output, axis=0)
            else :
                output = torch.zeros(200).to(device)
            outputs.append(output)
        outputs = torch.stack(outputs)

        train_tensor = torch.from_numpy(self.pre_w2v_emb.values).to(device)

        scores = torch.zeros([outputs.shape[0], train_tensor.shape[0]], dtype=torch.float64).to(device)
        for idx, vector in enumerate(outputs) :
            output = self.cos(vector.reshape(1, -1), train_tensor)
            scores[idx] = output

        scores = torch.sort(scores, descending=True)
        sorted_scores, sorted_idx = scores.values.cpu().numpy(), scores.indices.cpu().numpy()

        s = pd.DataFrame(sorted_scores, index=p_ids)
        i = pd.DataFrame(sorted_idx, index=p_ids).applymap(lambda x : self.pre_w2v_emb.index[x])        

        return pd.DataFrame([pd.Series(list(zip(i.loc[idx], s.loc[idx]))) for idx in p_ids], index=p_ids)        

    def _counting_question_data(self, q_songs, q_tags) :
        song_plylst_C = Counter()
        tag_song_C = Counter()

        for q_s in q_songs:
            song_plylst_C.update(self.song_plylst_dic[q_s])
        # 수록 tag에 대해
        for q_t in q_tags:
            tag_song_C.update(self.tag_song_dic[q_t])
        # 수록곡 수로 나눠서 비율로 계산
        for i, j in list(song_plylst_C.items()):
            if len(self.plylst_song_dic[i]) > 0:
                song_plylst_C[i] = (j / len(self.plylst_song_dic[i]))

        return song_plylst_C, tag_song_C
    
    def _check_question_status(self, q_songs, q_tags) :
        song_tag_status = 2
        if len(q_songs) == 0 and len(q_tags) == 0:
            song_tag_status = 0
        elif len(q_songs) <= 3:
            song_tag_status = 1

        return song_tag_status

    def _calc_scores(self, plylsts, scores, song_plylst_C, n_msp, n_mtp, q_songs, new_song_plylst_dict) :
        plylst_song_scores = defaultdict(float)
        plylst_tag_scores = defaultdict(float)
        
        # 3-1. plylst_song_scores 계산
        for idx, ms_p in enumerate(plylsts[0]):
            for song in self.plylst_song_dic[ms_p]:
                song_score = 0
                for q_s in q_songs:
                    try:
                        song_score += len(new_song_plylst_dict[q_s] & new_song_plylst_dict[song]) / len(new_song_plylst_dict[q_s])
                    except:
                        pass
                if song in self.freq_song:
                    plylst_song_scores[song] += song_plylst_C[ms_p] * song_score * scores[0][idx] * (n_msp - idx) * 4
                else:
                    plylst_song_scores[song] += song_plylst_C[ms_p] * song_score * scores[0][idx] * (n_msp - idx)

            for tag in self.plylst_tag_dic[ms_p]:
                plylst_tag_scores[tag] += scores[1][idx] * (n_msp - idx)

        # 3-2. plylst_tag_scores 계산
        for idx, mt_p in enumerate(plylsts[1]):
            for tag in self.plylst_tag_dic[mt_p]:
                plylst_tag_scores[tag] += scores[1][idx] * (n_mtp - idx)

            for song in self.plylst_song_dic[mt_p]:
                plylst_song_scores[song] += scores[1][idx]

        # 3-3. plylst_{song/tag}_scores 보정
        for idx, mt_p in enumerate(plylsts[2]):
            for song in self.plylst_song_dic[ms_p] :
                plylst_song_scores[song] += scores[2][idx] * (n_msp - idx)

            for tag in self.plylst_tag_dic[mt_p] :
                plylst_tag_scores[tag] += scores[2][idx] * (n_mtp - idx)

        plylst_song_scores = sorted(plylst_song_scores.items(), key = lambda x : x[1], reverse=True)
        plylst_tag_scores = sorted(plylst_tag_scores.items(), key = lambda x : x[1], reverse=True)
    
        return plylst_song_scores, plylst_tag_scores
        
    def _fill_no_data(self, plylst_song_scores, plylst_tag_scores) :
        # q_songs 새롭게 채워넣기 (원래는 song가 없지만 title_scores 기준 유사한 플레이리스트로부터 song 예측)
        pre_songs = [scores[0] for scores in plylst_song_scores][:200]
        pre_songs = pre_songs + remove_seen(pre_songs, self.song_popular)
        q_songs = pre_songs[:100]

        # q_tags 새롭게 채워넣기 (원래는 tag가 없지만 title_scores 기준 유사한 플레이리스트로부터 tag 예측)
        pre_tags = [scores[0] for scores in plylst_tag_scores][:20]
        pre_tags = pre_tags + remove_seen(pre_tags, self.tag_popular)
        q_tags = pre_tags[:10]

        return q_songs, q_tags

    def _exists_artist_filter(self, q_songs, song_tag_status, plylst_song_scores) :
        lt_song_art = []
        if len(q_songs) > 0 : # song 있을 때
            q_artists = []
            for w_song in q_songs:
                q_artists.extend(self.song_artist_dic[w_song])
        
            artist_counter = Counter(q_artists)
            artist_counter = sorted(artist_counter.items(), key=lambda x: x[1], reverse=True)
        
            if song_tag_status == 1:
                q_artists = [art[0] for art in artist_counter]
            else:
                q_artists = [x[0] for x in artist_counter if x[1] > 1]
        
            cand_ms = [scores[0] for scores in plylst_song_scores][(100 - len(q_artists)):1000]
            for cand in cand_ms:
                if q_artists == []:
                    break
                if cand in q_songs:
                    break
                for art in self.song_artist_dic[cand]:
                    if art in q_artists :
                        lt_song_art.append(cand)
                        q_artists.remove(art)
                        break
        
        return lt_song_art

    def inference(self, question_file_path, n_msp=50, n_mtp=90, save=True) :
        question_df = pd.read_json(question_file_path)

        auto_scores = self.similarity_by_auto(question_df, False)
        auto_gnr_scores = self.similarity_by_auto(question_df, True)
        w2v_scores = self.similarity_by_w2v(question_df)

        question_data = load_json(question_file_path)
        rec_list = []

        for q in question_data :
            q_id = q['id']
            q_songs = q['songs']
            q_tags = q['tags']
            
            song_plylst_C, tag_song_C = self._counting_question_data(q_songs, q_tags)

            song_tag_status = self._check_question_status(q_songs, q_tags)

            # Case 1: song과 tag가 둘 다 없는 경우
            if song_tag_status == 0:
                # plylst_ms / plylst_mt: title_scores 기준 유사한 플레이리스트 n_msp / n_mtp개
                plylst_ms, song_scores = most_similar_emb(q['id'], n_msp, w2v_scores)
                plylst_mt, tag_scores = most_similar_emb(q['id'], n_mtp, w2v_scores)
                plylst_add, add_scores = most_similar_emb(q['id'], n_mtp, auto_scores)

            # Case 2: song과 tag가 부족한 경우
            elif song_tag_status == 1 :
                plylst_ms, song_scores = most_similar_emb(q['id'], n_msp, auto_scores)
                plylst_mt, tag_scores = most_similar_emb(q['id'], n_mtp, w2v_scores)
                plylst_add, add_scores = most_similar_emb(q['id'], n_mtp, auto_gnr_scores)

            # Case 3: song과 tag가 충분한 경우
            else:
                plylst_ms, song_scores = most_similar_emb(q['id'], n_msp, auto_scores)
                plylst_mt, tag_scores = most_similar_emb(q['id'], n_mtp, auto_gnr_scores)
                plylst_add, add_scores = most_similar_emb(q['id'], n_mtp, w2v_scores)

            plylsts = [plylst_ms, plylst_mt, plylst_add]
            scores = [song_scores, tag_scores, add_scores]

            new_song_plylst_dict = defaultdict(set)
            for plylst in plylsts[0]:
                for _song in self.plylst_song_dic[plylst]:
                    new_song_plylst_dict[_song].add(plylst)

            plylst_song_scores, plylst_tag_scores = self._calc_scores(plylsts, scores, song_plylst_C, n_msp, n_mtp, q_songs, new_song_plylst_dict)

            if song_tag_status == 0 :
                q_songs, q_tags = self._fill_no_data(plylst_song_scores, plylst_tag_scores)

            lt_song_art = self._exists_artist_filter(q_songs, song_tag_status, plylst_song_scores)

            # 곡 추천
            if len(q_songs) > 0 : # song 있을 때
                song_similar = [scores[0] for scores in plylst_song_scores][:200]
            else : # song 없고, tag 있을 때
                song_similar = most_similar(tag_song_C, 200)

            ## 태그 추천
            tag_similar = [scores[0] for scores in plylst_tag_scores][:20]

            song_candidate = song_similar + remove_seen(song_similar, self.song_popular)
            tag_candidate = tag_similar + remove_seen(tag_similar, self.tag_popular)

            song_candidate = song_candidate[:100] if song_tag_status == 0 else remove_seen(q_songs, song_candidate)[:100]
            if len(lt_song_art) > 0:
                lt_song_art = [x for x in lt_song_art if x not in song_candidate]
                song_candidate[(100 - len(lt_song_art)):100] = lt_song_art

            tag_candidate = tag_candidate[:10] if song_tag_status == 0 else remove_seen(q_tags, tag_candidate)[:10]

            rec_list.append({"id": q_id, "songs": song_candidate, "tags": tag_candidate})

        if save == True:
            result_file_path = result_file_base.format(dt.datetime.now().strftime("%y%m%d-%H%M%S"))
            write_json(rec_list, result_file_path)
            print('Result file save to {}'.format(result_file_path))
        else :
            return rec_list

if __name__ == '__main__' :
    model = Recommender()

    rec_list = model.inference(question_file_path)
    print(pd.DataFrame(rec_list))
    