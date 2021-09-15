# Data library
import pandas as pd

# Models
from Models.recommender import Recommender

# Utils
from Utils.static import one_question_file_path
from Utils.file import load_json

if __name__ == '__main__' :
    print('Load Recommender Model')
    model = Recommender()

    print('Recommending...')
    rec_list = model.inference(load_json(one_question_file_path), save=False)
    print(rec_list)
    