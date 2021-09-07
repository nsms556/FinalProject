from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from Models.dataset import SongTagDataset
from Utils.file import load_json, write_json
from Utils.preprocessing import binary_songs2ids, binary_tags2ids


device = 'cuda' if torch.cuda.is_available() else 'cpu'

default_file_path = 'arena_data'
tag2id_file_path = f'{default_file_path}/tag2id_local_val.npy'
prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr2_local_val.npy'
id2tag_file_path = f'{default_file_path}/id2tag_local_val.npy'
id2song_file_path = f'{default_file_path}/id2freq_song_thr2_local_val.npy'
model_path = 'model/autoencoder_450_256_0.0005_0.2_2_local_val.pkl'
question_path = None
result_path = 'results/results.json'

def inference(q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019) :
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
    
    write_json(elements, result_path)

    return elements


id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
id2song_dict = dict(np.load(id2song_file_path, allow_pickle=True).item())

model = torch.load(model_path)

question_dataset = load_json(question_path)
question_dataset = SongTagDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)
question_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=8)

output = inference(question_dataloader, model, result_path, id2song_dict, id2tag_dict)