# Utils
from Utils.file import write_json

def inference(question_data_path, model, result_path) :
    element = model.inference(question_data_path, save=False)
            
    write_json(element, result_path)
    return element

def multi_lists(length, q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019):
    elements = []
    for i in range(length):
        elements.append(inference(q_dataloader, model, result_path, id2song_dict, id2tag_dict, num_songs=192019))
    
    return elements