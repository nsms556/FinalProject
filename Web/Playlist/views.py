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

from Playlist.recommend import inference

# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
def index(request):
    if request.method == 'GET':
        # json으로 나오는 추천 리스트 정보 
        # 추론 돌리고 나온 json 사용(현재 확인 편의상 simple.json 사용)
        json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
        with open(json_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        return JsonResponse({'playlists':json_data}, json_dumps_params={'ensure_ascii':True})
        
    elif request.method == 'POST':
        # 추천을 위한 정보 json으로 받아오기
        body = json.loads(request.body.decode('utf-8'))
        return JsonResponse({'success': True, 'input':body, 'datatype':str(type(body))}, json_dumps_params={'ensure_ascii': True})
    

# '/detail' 플레이리스트 내부 곡 정보
def detail(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
    with open(json_path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        playlist_songs = []

        for i in range(len(json_data)):
            songs = []
            for song in json_data[i]['songs']:
                cur.execute(f'select song_name, artist_name_basket from song_meta where id={song}')
                rows = cur.fetchall()
                for row in rows:
                    songs.append(row)
            playlist_songs.append(songs)

    return JsonResponse({'playlist_songs':playlist_songs}, json_dumps_params={'ensure_ascii':True})

@method_decorator(csrf_exempt, name='dispatch')
def show_inference(request):
    if request.method == 'GET':
        return JsonResponse({'success':True}, json_dumps_params={'ensure_ascii':True})

    elif request.method == 'POST':
        # 추천을 위한 정보 json으로 받고 추천
        body = json.loads(request.body.decode('utf-8'))
        
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

        question_dataset = body
        question_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=8)

        output = inference(question_dataloader, model, result_path, id2song_dict, id2tag_dict)
        return JsonResponse({'success': True, 'output': output}, json_dumps_params={'ensure_ascii': True})
    