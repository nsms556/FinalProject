import torch
from tqdm import tqdm

from Utils.preprocessing import binary_songs2ids, binary_tags2ids
from Utils.file import write_json, load_json, check


def inference(q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019) :
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            for idx, (_id, _data) in tqdm(enumerate(q_dataloader), desc='testing...') :
                with torch.no_grad() :
                    _data = _data.to(device)
                    output = model(_data)

                songs_input, tags_input = torch.split(_data, num_songs, dim=1)
                songs_output, tags_output = torch.split(output, num_songs, dim=1)

                songs_ids = binary_songs2ids(songs_input, songs_output, id2song_dict)
                tags_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

                _id = list(map(int, _id))
                element = {'id':_id, 'songs':list(songs_ids), 'tags':tags_ids}
            
            write_json(element, result_path)

            return element

def multi_lists(length, q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019):
    elements = []
    for i in range(length):
        elements.append(inference(q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019))
    
    return elements