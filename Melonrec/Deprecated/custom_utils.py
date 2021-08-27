import os
from collections import defaultdict
from tqdm import tqdm

import torch

from utils.arena_util import write_json
from utils.evaluate import ArenaEvaluator
from utils.data_util import binary_songs2ids, binary_tags2ids


def tmp_file_remove(file_path) :
  if os.path.exists(file_path) :
    os.remove(file_path)

def mid_check(q_dataloader, model, tmp_result_path, answer_file_path, id2song_dict, id2tag_dict, is_cuda, num_songs) :
    evaluator = ArenaEvaluator()
    device = 'cuda' if is_cuda else 'cpu'

    tmp_file_remove(tmp_result_path)

    elements =[]
    for idx, (_id, _data) in tqdm(enumerate(q_dataloader), desc='testing...') :
        with torch.no_grad() :
            _data = _data.to(device)
            output = model(_data)

        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
        songs_output, tags_output = torch.split(output, num_songs, dim=1)

        songs_ids = binary_songs2ids(songs_input, songs_output, id2song_dict)
        tags_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

        _id = list(map(int, _id))
        for i in range(len(_id)) :
            element = {'id':_id[i], 'songs':list(songs_ids[i]), 'tags':tags_ids[i]}
            elements.append(element)
    
    write_json(elements, tmp_result_path)
    evaluator.evaluate(answer_file_path, tmp_result_path)
    os.remove(tmp_result_path)

## 빠른 접근을 위한 Dictionary 생성
def DicGenerator(train, song_meta):
    # key: song / value: issue_date
    song_issue_dic = defaultdict(lambda: '')

    for i in range(len(song_meta)):
        song_issue_dic[song_meta[i]['id']] = song_meta[i]['issue_date']

    # key: song / value: artist_id_basket
    song_artist_dic = defaultdict(lambda: [])

    for i in range(len(song_meta)):
        lt_art_id = song_meta[i]['artist_id_basket']
        song_artist_dic[song_meta[i]['id']] = lt_art_id

    # key: song / value: playlist
    song_plylst_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        for t_s in train[i]['songs']:
            song_plylst_dic[t_s] += [train[i]['id']]

    # key: song / value: tag
    song_tag_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        for t_s in train[i]['songs']:
            song_tag_dic[t_s] += train[i]['tags']

    # key: plylst / value: song
    plylst_song_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        plylst_song_dic[train[i]['id']] += train[i]['songs']

    # key: plylst / value: tag
    plylst_tag_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        plylst_tag_dic[train[i]['id']] += train[i]['tags']

    # key: tag / value: plylst
    tag_plylst_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        for t_q in train[i]['tags']:
            tag_plylst_dic[t_q] += [train[i]['id']]

    # key: tag / value: song
    tag_song_dic = defaultdict(lambda: [])

    for i in range(len(train)):
        for t_q in train[i]['tags']:
            tag_song_dic[t_q] += train[i]['songs']

    return song_plylst_dic, song_tag_dic, plylst_song_dic, plylst_tag_dic, tag_plylst_dic, tag_song_dic, song_issue_dic, song_artist_dic