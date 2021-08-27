from django.shortcuts import render
import os
import json
from django.conf import settings
import sqlite3
from django.http import HttpResponse, JsonResponse

# Create your views here.
def index(request):
    # json으로 나오는 추천 리스트 정보 
    json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
    with open(json_path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        playlists = []
        for i in range(len(json_data)):
            playlists.append(json_data[i])

    return render(request, 'index.html', {'playlists':playlists})

def detail(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    json_path = os.path.join(settings.BASE_DIR, 'Playlist/results/simple.json')
    with open(json_path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        playlists = []
        playlist_songs = []

        for i in range(len(json_data)):
            playlists.append(json_data[i])
            songs = []
            for song in json_data[i]['songs']:
                cur.execute(f'select song_name, artist_name_basket from song_meta where id={song}')
                rows = cur.fetchall()
                for row in rows:
                    songs.append(row)
            playlist_songs.append(songs)

    return JsonResponse({'playlists': playlists, 'playlist_songs':playlist_songs}, json_dumps_params={'ensure_ascii':True})