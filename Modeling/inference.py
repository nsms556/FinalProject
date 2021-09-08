# Data library
import pandas as pd

# Models
from Models.recommender import Recommender

# Utils
from Utils.static import one_question_file_path


if __name__ == '__main__' :
    print('Load Recommender Model')
    model = Recommender()

    print('Recommending...')
    rec_list = model.inference(one_question_file_path, save=False)
    print(pd.DataFrame(rec_list))
    