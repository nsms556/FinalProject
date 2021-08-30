import os
import json
from datetime import datetime

from collections import defaultdict, Counter


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def most_popular(plylsts, key, n):
    counter = Counter()
    for plylst in plylsts:
        counter.update(plylst[key])
    counter_top_n = counter.most_common(n)
    song_tags, counts = zip(*counter_top_n)
    return counter_top_n, song_tags
    
# 0. Datasets
if not (os.path.isdir('Datasets/')):
    os.makedirs('Datasets/')

train = load_json('Datasets/train.json')
val = load_json('Datasets/val.json')
test = load_json('Datasets/test.json')
# train = train + val + test

# 1. Rank
print('Rank popular songs/tags...')
data_by_yearmonth = defaultdict(dict)
for plylst in train:
    _datetime = datetime.fromisoformat(plylst['updt_date'])
    YYYY, M = _datetime.year, _datetime.month
    try:
        data_by_yearmonth[YYYY][M].append(plylst)
    except KeyError:
        data_by_yearmonth[YYYY] = defaultdict(list)
# print(data_by_yearmonth[2012][12])

#     try:
#         data_by_yearmonth[q['updt_date'][0:4]].append(q)
#     except:
#         pass
#     try:
#         data_by_yearmonth[q['updt_date'][0:7]].append(q)
#     except:
#         pass
# data_by_yearmonth = dict(data_by_yearmonth)

# 2.
popular_songs = 100
popular_tags = 50
most_popular_results = {}
songs_mp_counter, most_popular_results['songs'] = most_popular(train, "songs", popular_songs)
tags_mp_counter, most_popular_results['tags'] = most_popular(train, "tags", popular_tags)

# for y in data_by_yearmonth.keys():
#     _, most_popular_results['songs' + y] = most_popular(data_by_yearmonth[y], "songs", popular_songs)
#     _, most_popular_results['tags' + y] = most_popular(data_by_yearmonth[y], "tags", popular_tags)

# print('split title into words...')
# all_word_list = []
# for t in tags_mp_counter.most_common():  # use tags as words dictionary
#     if t[1] >= 5 and len(t[0]) > 1:
#         all_word_list.append(t)
# for q in train:
#     q['title_words'] = title_into_words(q['plylst_title'])
# for q in test:
#     q['title_words'] = title_into_words(q['plylst_title'])
#
# print('write train matrix...')
# playlist_song_train_matrix = []
# p_encode, s_encode, p_decode, s_decode = {}, {}, {}, {}
# playlist_idx = 0
# song_idx = 0
# for q in train:
#     if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
#         p_encode[q['id']] = playlist_idx
#         for s in q['songs']:
#             if s not in s_encode.keys():
#                 s_encode[s] = song_idx
#                 song_idx += 1
#             playlist_song_train_matrix.append([playlist_idx, s_encode[s]])
#         playlist_idx += 1
# s_decode['@tag_start_idx'] = song_idx
# for q in train:
#     if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
#         for s in q['tags']:
#             if s not in s_encode.keys():
#                 s_encode[s] = song_idx
#                 song_idx += 1
#             playlist_song_train_matrix.append([p_encode[q['id']], s_encode[s]])
# s_decode['@tag_title_start_idx'] = song_idx
# for q in train:
#     if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
#         for s in q['title_words']:
#             if '!title_' + str(s) not in s_encode.keys():
#                 s_encode['!title_' + str(s)] = song_idx
#                 song_idx += 1
#             playlist_song_train_matrix.append([p_encode[q['id']], s_encode['!title_' + str(s)]])
# playlist_song_train_matrix = np.array(playlist_song_train_matrix)
# playlist_song_train_matrix = coo_matrix((np.ones(playlist_song_train_matrix.shape[0]),
#                                          (playlist_song_train_matrix[:, 0], playlist_song_train_matrix[:, 1])),
#                                         shape=(playlist_idx, song_idx))
# save_npz('data/playlist_song_train_matrix.npz', playlist_song_train_matrix)
# for s in s_encode.keys():
#     s_decode[s_encode[s]] = s
# pickle_dump(s_decode, 'data/song_label_decoder.pickle')
# pickle_dump(p_encode, 'data/playlist_label_encoder.pickle')
#
# title_words_mp_counter, _ = most_popular(train, "title_words", 50)
#
# print('write test item indices...')
# for q in test:
#     if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
#         if np.mean([songs_mp_counter[i] for i in q['songs']] + [tags_mp_counter[i] for i in q['tags']] + [
#             title_words_mp_counter[i] for i in q['title_words']]) > 1:
#             items = [s_encode[s] for s in q['songs'] + q['tags']]
#             try:
#                 for s in q['title_words']:
#                     if '!title_' + str(s) in s_encode.keys():
#                         items.append(s_encode['!title_' + str(s)])
#             except KeyError:
#                 q['title_words'] = []
#             q['items'] = items
#
#     if 'songs' + q['updt_date'][0:7] in most_popular_results.keys():
#         q['songs_mp'] = (remove_seen(q['songs'], most_popular_results['songs' + q['updt_date'][0:7]] + remove_seen(
#             most_popular_results['songs' + q['updt_date'][0:7]], most_popular_results['songs'])))[:100]
#         q['tags_mp'] = (remove_seen(q['tags'], most_popular_results['tags' + q['updt_date'][0:7]] + remove_seen(
#             most_popular_results['tags' + q['updt_date'][0:7]], most_popular_results['tags'])))[:10]
#     elif 'songs' + q['updt_date'][0:4] in most_popular_results.keys():
#         q['songs_mp'] = (remove_seen(q['songs'], most_popular_results['songs' + q['updt_date'][0:4]] + remove_seen(
#             most_popular_results['songs' + q['updt_date'][0:4]], most_popular_results['songs'])))[:100]
#         q['tags_mp'] = (remove_seen(q['tags'], most_popular_results['tags' + q['updt_date'][0:4]] + remove_seen(
#             most_popular_results['tags' + q['updt_date'][0:4]], most_popular_results['tags'])))[:10]
#     else:
#         q['songs_mp'] = remove_seen(q['songs'], most_popular_results['songs'][:100])
#         q['tags_mp'] = remove_seen(q['tags'], most_popular_results['tags'][:10])
#
# write_json(test, 'data/test_items.json')