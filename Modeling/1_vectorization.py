from Utils import tags_ids_convert, save_freq_song_id_dict


if __name__ == "__main__":
    # 1. Embedding 준비
    
    # 1-1. tag2id & id2tag
    # Autoencoder의 input: song, tag binary vector의 concatenate, tags는 str이므로 id로 변형할 필요 있음
    tag2id_file_path = f'{default_file_path}/tag2id_{model_postfix}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{model_postfix}.npy'
    
    # 관련 데이터들이 없으면 default file path에 새로 만들음
    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_data, tag2id_file_path, id2tag_file_path)

    # 1-2. freq_song2id & id2freq_song
    # Song이 너무 많기 때문에 frequency에 기반하여 freq_thr번 이상 등장한 곡들만 남김, 남은 곡들에게 새로운 id 부여
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr{freq_thr}_{model_postfix}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr{freq_thr}_{model_postfix}.npy'

    # 관련 데이터들이 없으면 default file path에 새로 만들음
    if not (os.path.exists(prep_song2id_file_path) & os.path.exists(id2prep_song_file_path)):
        save_freq_song_id_dict(train_data, freq_thr, default_file_path, model_postfix)

    '''
    train_dataset = SongTagDataset(train_data, tag2id_file_path, prep_song2id_file_path)
    if question_data is not None:
        question_dataset = SongTagDataset(question_data, tag2id_file_path, prep_song2id_file_path)

    model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}_{}.pkl'. \
        format(H, batch_size, learning_rate, dropout, freq_thr, model_postfix)

    train(train_dataset, model_file_path, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path)
    '''
    
    # 2. Song One-hot Vector
    
    
    # 3. Tag One-hot Vector
    