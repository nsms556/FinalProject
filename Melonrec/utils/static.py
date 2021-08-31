train_file_base = '{}/train.json'
val_file_base = '{}/val.json'
question_file_base = '{}/test.json'
genre_meta_file = 'Lib/genre_gn_all.json'
song_meta_file = 'Lib/song_meta.json'

tag2id_file_path = 'Lib/tag2id.npy'
id2tag_file_path = 'Lib/id2tag.npy'
song2id_file_path = 'Lib/song2id.npy'
id2song_file_path = 'Lib/id2song.npy'

autoencoder_model_path = 'Weights/autoencoder_{}_{}_{}_{}_{}_{}.pkl'
tokenize_input_file = 'Weights/tokenizer_input.txt'
vectorizer_model_path = 'Weights/vectorizer.model'

tmp_results_path = 'results/tmp_results.json'
temp_fn = 'Arena_data/answers/temp.json'

plylst_emb_path = 'Lib/plylst_emb.npy'
plylst_emb_gnr_path = 'Lib/plylst_emb_gnr.npy'
plylst_w2v_emb_path = 'Lib/plylst_w2v_emb.npy'

autoencoder_score_file_path = 'Scores/scores_bias_without_gnr'
autoencoder_gnr_score_file_path = 'Scores/scores_bias_with_gnr'
word2vec_score_file_path = 'Scores/scores_title'

result_file_base = 'Results/results_{}_{}.json'