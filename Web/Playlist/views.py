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

from Utils.file import write_json
from Utils.static import question_file_base, result_file_base
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
    
    # 정보 json으로 받고 추천
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            
            t = dt.datetime.now().strftime("%y%m%d-%H%M%S")
            write_json(body, question_file_base.format(t))
            output = inference(question_file_base.format(t), model, result_file_base.format(t))

            '''
                question_dataset = body
                question_dataset = SongTagDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)
                question_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=8)

                output = inference(question_dataloader, model, result_path, id2song_dict, id2tag_dict)
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

            return JsonResponse({'song_list': song_list}, json_dumps_params={'ensure_ascii': True})
        except (KeyError, JSONDecodeError, ValueError) as e:
            pass