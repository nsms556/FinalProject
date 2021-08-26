train_file_base = '{}/train.json'
val_file_base = '{}/val.json'
qustion_file_base = '{}/test.json'
genre_file_path = 'res/genre_gn_all.json'
song_file_path = 'res/song_meta.json'

tag2id_file_base = '{}/tag2id_{}.npy'
id2tag_file_base = '{}/id2tag_{}.npy'
song2id_file_base = '{}/song2id_thr{}_{}.npy'
id2song_file_base = '{}/id2song_thr{}_{}.npy'

autoencoder_model_base = 'model/autoencoder_{}_{}_{}_{}_{}_{}.pkl'
tokenize_input_file_base = 'model/tokenizer_input_{}.txt'

tmp_results_path = 'results/tmp_results.json'
temp_fn = 'arena_data/answers/temp.json'

plylst_emb_base = '{}/plylst_emb.npy'
plylst_emb_gnr_base = '{}/plylst_emb_gnr.npy'
plylst_w2v_emb_base = '{}/plylst_w2v_emb.npy'

autoencoder_score_file_base = 'scores/{}_scores_bias_{}'
autoencoder_gnr_score_file_base = 'scores/{}_scores_bias_{}_gnr'
word2vec_score_file_base = 'scores/{}_scores_title_{}'

result_file_base = 'results/results_{}_{}.json'