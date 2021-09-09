# Paths
train_file_path = 'Data/train.json'
answer_file_path = 'Data/answer.json'
question_file_path = 'Data/question.json'
genre_meta_file_path = 'Lib/genre_gn_all.json'
song_meta_file_path = 'Lib/song_meta.json'

tag2id_file_path = 'Lib/tag2id.npy'
id2tag_file_path = 'Lib/id2tag.npy'
song2id_file_path = 'Lib/song2id.npy'
id2song_file_path = 'Lib/id2song.npy'

autoencoder_model_path = 'Weights/autoencoder_model.pth'
autoencoder_encoder_layer_path = 'Weights/autoencoder_weights.pth'
vectorizer_weights_path = 'Weights/w2v.weights'

plylst_emb_path = 'Lib/plylst_emb.npy'
plylst_emb_gnr_path = 'Lib/plylst_emb_gnr.npy'
plylst_w2v_emb_path = 'Lib/plylst_w2v_emb.npy'

result_file_base = 'Results/results_{}.json'


# Train Only
tokenize_input_file_path = 'Weights/tokenizer_input.txt'

tmp_results_path = 'results/tmp_results.json'
temp_fn = 'Arena_data/answers/temp.json'

# Inference Test
one_question_file_path = 'Data/one_question.json'

# Deprecate
autoencoder_score_file_path = 'Scores/scores_bias_without_gnr.npy'
autoencoder_gnr_score_file_path = 'Scores/scores_bias_with_gnr.npy'
word2vec_score_file_path = 'Scores/scores_title.npy'