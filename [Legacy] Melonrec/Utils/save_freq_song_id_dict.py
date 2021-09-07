def save_freq_song_id_dict(train, thr, default_file_path, model_postfix):
    song_counter = collections.Counter()
    for plylst in train:
        song_counter.update(plylst['songs'])

    selected_songs = []
    song_counter = list(song_counter.items())
    for k, v in song_counter:
        if v > thr:
            selected_songs.append(k)

    print(f'{len(song_counter)} songs to {len(selected_songs)} songs')

    freq_song2id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix}', freq_song2id)
    print(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix} is created')
    id2freq_song = {v: k for k, v in freq_song2id.items()}
    np.save(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix}', id2freq_song)
    print(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix} is created')