from django.shortcuts import render
import os
import json
from django.conf import settings
import sqlite3
from django.http import HttpResponse, JsonResponse

# Create your views here.
def index(request):
    # json으로 나오는 추천 리스트 정보 
    # 추론 돌리고 나온 results_210812-170628_test.json 사용
    # (현재 확인 편의상 simple.json 사용)
    json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
    with open(json_path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    return JsonResponse({'playlists':json_data}, json_dumps_params={'ensure_ascii':True})

# [/detail] 플레이리스트 내부 곡 정보
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