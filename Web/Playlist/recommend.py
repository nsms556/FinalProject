# Utils
from Utils.file import write_json
from Utils.preprocessing import remove_seen

def inference(question_data, model, result_path) :
    question_data = question_data[0]

    like_data = [question_data['like']]
    dislike_data = [question_data['dislike']]
    
    like = model.inference(like_data, save=False)[0]
    dislike = model.inference(dislike_data, save=False)[0]
    
    true_like_songs = remove_seen(dislike['songs'], like['songs'])
    maybe_like_songs = list(set(like['songs']) & set(dislike['songs']))

    true_like_tags = remove_seen(dislike['tags'], like['tags'])
    maybe_like_tags = list(set(like['tags']) & set(dislike['tags']))
    
    true_like = {'id':like['id'], 'songs':true_like_songs, 'tags':true_like_tags}
    maybe_like = {'id':like['id'], 'songs':maybe_like_songs, 'tags':maybe_like_tags}

    write_json(true_like, result_path)
    return true_like, maybe_like

def multi_lists(length, q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019):
    elements = []
    for i in range(length):
        elements.append(inference(q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019))
    
    return elements