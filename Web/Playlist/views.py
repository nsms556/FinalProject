from django.shortcuts import render
import os
import json
from django.conf import settings
import sqlite3
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

from Playlist.recommend import inference
from Utils.dataset import SongTagDataset

#region [MODEL]
default_file_path = 'checkpoint/arena_data'
tag2id_file_path = f'{default_file_path}/tag2id_local_val.npy'
prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr2_local_val.npy'
id2tag_file_path = f'{default_file_path}/id2tag_local_val.npy'
id2song_file_path = f'{default_file_path}/id2freq_song_thr2_local_val.npy'
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'Playlist/model/autoencoder_450_256_0.0005_0.2_2_local_val.pkl')
question_path = None
result_path = 'results/results.json'

id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
id2song_dict = dict(np.load(id2song_file_path, allow_pickle=True).item())

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()
#endregion

# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
def index(request):
    if request.method == 'GET':
        # 플레이리스트 홈화면
        json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
        with open(json_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        return JsonResponse({'playlists':json_data}, json_dumps_params={'ensure_ascii':True})
    
    # 노래 검색
    elif request.method == 'POST':
        body = json.loads(request.body.decode('utf-8'))
        song_title = body['song_title']
        artist = body['artist']

        conn = sqlite3.connect('data.db')
        cur = conn.cursor()
        if song_title and artist:
            cur.execute(f"select id, song_name, artist_name_basket, album_name, album_id, issue_date \
                    from song_meta \
                    where song_name LIKE '%{song_title}%' and artist_name_basket LIKE '%{artist}%'")
        elif song_title and not artist:
            cur.execute(f"select id, song_name, artist_name_basket, album_name, album_id, issue_date \
                    from song_meta \
                    where song_name LIKE '%{song_title}%'")
        else:
            cur.execute(f"select id, song_name, artist_name_basket, album_name, album_id, issue_date \
                    from song_meta \
                    where artist_name_basket LIKE '%{artist}%'")
        output = []
        rows = cur.fetchall()
        for row in rows:
            output.append(row)
        return JsonResponse({'success': True, 'output':output}, json_dumps_params={'ensure_ascii': True})
    

# '/detail' 플레이리스트 내부 곡 정보
@method_decorator(csrf_exempt, name='dispatch')
def detail(request):
    return JsonResponse({'success': True}, json_dumps_params={'ensure_ascii': True})
    
@method_decorator(csrf_exempt, name='dispatch')
def show_inference(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    if request.method == 'GET':
        return JsonResponse({'success':True}, json_dumps_params={'ensure_ascii':True})

    if request.method == 'POST':
        # 추천을 위한 정보 json으로 받고 추천
        body = json.loads(request.body.decode('utf-8'))
        question_dataset = body
        question_dataset = SongTagDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)
        question_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=8)

        output = inference(question_dataloader, model, result_path, id2song_dict, id2tag_dict)
        song_names = []

        for song in output['songs'][0]:
            cur.execute(f'select song_name, artist_name_basket from song_meta where id={song}')
            rows = cur.fetchall()
            for row in rows:
                song_names.append(row)

        return JsonResponse({'song_list': song_names}, json_dumps_params={'ensure_ascii': True})