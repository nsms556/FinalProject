import os
import argparse
import torch
import torch.nn as nn
from utils.data_util import tags_ids_convert, save_freq_song_id_dict
from utils.train import train


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


def train_AutoEncoder():
    pass


def get_plylsts_embeddings(_model_file_path, _submit_type, genre=False):
    if _submit_type == 'val':
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
        val_file_path = 'res/val.json'
        train_dataset = load_json(train_file_path)
    elif _submit_type == 'test':
        default_file_path = 'res'
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
        val_file_path = 'res/val.json'
        train_dataset = load_json(train_file_path) + load_json(val_file_path)
    elif _submit_type == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        default_file_path = f'{default_file_path}/orig'
        train_dataset = load_json(train_file_path)

    tag2id_file_path = f'{default_file_path}/tag2id_{_submit_type}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{_submit_type}.npy'
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr2_{_submit_type}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr2_{_submit_type}.npy'

    if genre:
        train_dataset = SongTagGenreDataset(train_dataset, tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagGenreDataset(load_json(question_file_path), tag2id_file_path, prep_song2id_file_path)
    else:
        train_dataset = SongTagDataset(train_dataset, tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagDataset(load_json(question_file_path), tag2id_file_path, prep_song2id_file_path)

    plylst_embed_weight = []
    plylst_embed_bias = []

    model_file_path = _model_file_path

    model = torch.load(model_file_path)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'encoder.1.weight':
                plylst_embed_weight = param.data
            elif name == 'encoder.1.bias':
                plylst_embed_bias = param.data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=4)
    question_loader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plylst_emb_with_bias = dict()

    if genre:
        for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(train_loader, desc='get train vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(question_loader, desc='get question vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]
    else:
        for idx, (_id, _data) in enumerate(tqdm(train_loader, desc='get train vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        for idx, (_id, _data) in enumerate(tqdm(question_loader, desc='get question vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]
    return plylst_emb_with_bias


if __name__ == "main":
    # 0. 하이퍼 파라미터 입력
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=41)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=20)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-mode', type=int, help="local_val: 0, val: 1, test: 2", default=2)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    freq_thr = args.freq_thr
    mode = args.mode
    if mode == 0:
        model_postfix = 'local_val'
    elif mode == 1:
        model_postfix = 'val'
    elif mode == 2:
        model_postfix = 'test'

    # 0. datasets
    default_datasets_path = './datasets'

    train_file_path = f'{default_datasets_path}/small_datasets/train.json'
    question_file_path = f'{default_datasets_path}/small_datasets/questions/val.json'
    answer_file_path = f'{default_datasets_path}/small_datasets/answers/val.json'

    train_data = load_json(train_file_path)
    question_data = load_json(question_file_path)

    # 1. AutoEncoder
    default_files_path = 'models/files'
    default_models_path = './models'

    # AutoEncoder/Song One-hot Vector 생성을 위한 파일 생성
    tag2id_file_path = f'{default_files_path}/tag2id.npy'
    id2tag_file_path = f'{default_files_path}/id2tag.npy'
    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_data, tag2id_file_path, id2tag_file_path)

    # AutoEncoder/Tag One-hot Vector 생성을 위한 파일 생성
    prep_song2id_file_path = f'{default_files_path}/freq_song2id_thr{freq_thr}_{model_postfix}.npy'
    id2prep_song_file_path = f'{default_files_path}/id2freq_song_thr{freq_thr}_{model_postfix}.npy'
    if not (os.path.exists(prep_song2id_file_path) & os.path.exists(id2prep_song_file_path)):
        save_freq_song_id_dict(train_data, freq_thr, default_files_path, model_postfix)

    # AutoEncoder train
    model_file_path = f'./models/autoencoder_{H}_{batch_size}_{learning_rate}_{dropout}_{freq_thr}_{model_postfix}.pkl'
    train_AutoEncoder(train_dataset, model_file_path, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path)

    # get_plylsts_embeddings
    get_plylsts_embeddings(_model_file_path, _submit_type, genre=False)