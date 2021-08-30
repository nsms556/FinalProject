class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        encoder_layer = nn.Linear(D_in, H, bias=True)
        decoder_layer = nn.Linear(H, D_out, bias=True)
        
        torch.nn.init.xavier_uniform_(encoder_layer.weight)
        torch.nn.init.xavier_uniform_(decoder_layer.weight)
        
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            encoder_layer,
            nn.BatchNorm1d(H),
            nn.LeakyReLU())
        self.decoder = nn.Sequential(
            decoder_layer,
            nn.Sigmoid())
    
    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder
    
    
def train_autoencoder(self, train_dataset, question_dataset, model_file_path, id2tag_file_path, id2prep_song_file_path,
          answer_file_path):
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('device: CUDA')
    else:
        device = torch.device('cpu')
        print('device: CPU')
    
    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())
    
    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags
    
    # hyper parameters
    D_in = D_out = num_songs + num_tags
    
    #
    q_data_loader = None
    check_every = 5
    tmp_result_file_path = 'results/tmp_results.json'
    evaluator = ArenaEvaluator()
    if question_dataset is not None:
        q_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)
    
    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    
    #
    try:
        model = torch.load(model_file_path)
        print("Model loaded! :)")
    except FileNotFoundError:
        print("Model does not exists... :(")
    
    #
    # temp_fn = 'arena_data/answers/temp.json'
    temp_fn = ''
    if os.path.exists(temp_fn):
        os.remove(temp_fn)
    
    # train
    for epoch in range(epochs):
        print()
        print(f'Epoch: {epoch}')
        running_loss = 0.0
        for idx, (_id, _data) in enumerate(tqdm(data_loader, desc='Training now...')):
            _data = _data.to(device)
            
            optimizer.zero_grad()
            output = model(_data)
            loss = loss_func(output, _data)
            loss.backword()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Loss: {epoch} {epoch / (epoch * 100)} {running_loss}')
        
        torch.save(model, model_file_path)
        
        #
        if mode == 0:
            if epoch % check_every == 0:
                if os.path.exists(tmp_result_file_path):
                    os.remove(tmp_result_file_path)
                elements = []
                for idx, (_id, _data) in enumerate(tqdm(q_data_loader, desc='testing...')):
                    with torch.no_grad():
                        _data = _data.to(device)
                        output = model(_data)
                        
                        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
                        songs_output, tags_output = torch.split(output, num_songs, dim=1)
                        
                        songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
                        tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)
                        
                        _id = list(map(int, _id))
                        for i in range(len(_id)):
                            element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                            elements.append(element)
                
                write_json(elements, tmp_result_file_path)
                evaluator.evaluate(answer_file_path, tmp_result_file_path)
                os.remove(tmp_result_file_path)