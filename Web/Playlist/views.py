import os
import logging
import json
from json.decoder import JSONDecodeError
import datetime as dt

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

import sqlite3
import webbrowser
import urllib.request
import urllib.parse

from Utils.static import result_file_base
from Models.recommender import Recommender

from Playlist.recommend import inference

#region [MODEL]
model = Recommender()
#endregion

# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
def index(request):
    if request.method == 'GET':
        # 플레이리스트 홈화면
        return JsonResponse({'status':"home page"}, json_dumps_params={'ensure_ascii':True})
    
    # 노래 검색
    elif request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            type = body['type']
            word = body['word']

            conn = sqlite3.connect('data.db')
            cur = conn.cursor()
            
            if type == "song_name":
                cur.execute(f"select id, song_name, artist_name_basket, album_name, album_id, issue_date \
                        from song_meta \
                        where song_name LIKE '%{word}%'")
            else:
                cur.execute(f"select id, song_name, artist_name_basket, album_name, album_id, issue_date \
                        from song_meta \
                        where artist_name_basket LIKE '%{word}%'")
            output = []
            rows = cur.fetchall()
            for row in rows:
                content = {
                        'song_id': row[0],
                        'song_name': row[1],
                        'artist': row[2],
                        'album_id': row[3],
                        'album_name': row[4],
                        'issue_date': row[5]
                    }
                output.append(content)
            return JsonResponse({'output':output}, json_dumps_params={'ensure_ascii': True})
        except KeyError:
            return # redirect home
    

# '/detail' 곡 세부정보
@method_decorator(csrf_exempt, name='dispatch')
def detail(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))

            playlink = "https://www.youtube.com/results?search_query="
            encode = urllib.parse.quote_plus(body['artist_name']) + "+" + "+".join(urllib.parse.quote_plus(body['song_name']).split())
            playlink += encode
            webbrowser.open(playlink)
            return JsonResponse({'success': True, 'playlink':playlink}, json_dumps_params={'ensure_ascii': True})

        except (KeyError, JSONDecodeError, ValueError) as e:
            return JsonResponse({'success':False}, json_dumps_params={'ensure_ascii': True})

    
@method_decorator(csrf_exempt, name='dispatch')
def show_inference(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()

    if request.method == 'GET':
        return JsonResponse({'success':True}, json_dumps_params={'ensure_ascii':True})
    
    # 정보 json으로 받고 추천
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            output = inference(body, model, result_file_base.format(dt.datetime.now().strftime("%y%m%d-%H%M%S")))
            
            '''
                question_dataset = body
                question_dataset = SongTagDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)
                question_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=8)

                output = inference(question_dataloader, model, result_path, id2song_dict, id2tag_dict)
            '''
            '''
            song_list = []

            for song in output['songs'][0]:
                cur.execute(f"select id, song_name, artist_name_basket, album_id, album_name, issue_date,\
                    song_gn_gnr_basket, song_gn_dtl_gnr_basket\
                    from song_meta\
                    where id={song}")
                rows = cur.fetchall()
                for row in rows:
                    content = {
                        'song_id': row[0],
                        'song_name': row[1],
                        'artist': row[2],
                        'album_id': row[3],
                        'album_name': row[4],
                        'issue_date': row[5],
                        'gn_gnr' : row[6],
                        'dtl_gnr': row[7]
                    }
                    song_list.append(content)
            '''
            return JsonResponse({'ouput': output}, json_dumps_params={'ensure_ascii': True})
        except (KeyError, JSONDecodeError, ValueError) as e:
            return JsonResponse({'ouput': e}, json_dumps_params={'ensure_ascii': True})