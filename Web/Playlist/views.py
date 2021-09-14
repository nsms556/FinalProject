import os
import logging
import json
from json.decoder import JSONDecodeError
import datetime as dt

from django.urls import reverse
from django.shortcuts import redirect, render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

import sqlite3
import webbrowser
import urllib.request
import urllib.parse

from Utils.static import result_file_base
from Utils.file import load_json
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
        return JsonResponse({'success':True, 'status':"home page"}, json_dumps_params={'ensure_ascii':True})
    
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
            return JsonResponse({'success':True, 'output':output}, json_dumps_params={'ensure_ascii': True})
        except KeyError:
            return # redirect home
    

# '/detail' 노래 재생(유튜브 검색)
@method_decorator(csrf_exempt, name='dispatch')
def detail(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))

            playlink = "https://www.youtube.com/results?search_query="
            encode = f"official+{urllib.parse.quote_plus(body['artist_name'])}+{'+'.join(urllib.parse.quote_plus(body['song_name']).split())}"
            playlink += encode
            webbrowser.open(playlink)
            return JsonResponse({'success': True, 'playlink':playlink}, json_dumps_params={'ensure_ascii': True})

        except (KeyError, JSONDecodeError, ValueError) as e:
            return JsonResponse({'success':False}, json_dumps_params={'ensure_ascii': True})

    
@method_decorator(csrf_exempt, name='dispatch')
def show_inference(request):
    if request.method == 'GET':
        return JsonResponse({'success':True}, json_dumps_params={'ensure_ascii':True})
    
    if request.method == 'POST':
        try:
            conn = sqlite3.connect('data.db')
            cur = conn.cursor()

            body = json.loads(request.body.decode('utf-8'))
            u_id = request.session['u_id']

            like = body[0]["like"]
            dislike = body[0]["dislike"]
            print(like)
            print(dislike)
            for tag in like["tags"]:
                query = f"""
                    INSERT into usr_gnr (u_id, gnr_name, isLike) values({u_id}, {tag}, 1)
                    """
                cur.execute(query)
                conn.commit()
            for song_id in like["songs"]:
                query = f"""
                    INSERT into usr_songs (u_id, song_id, isLike) values({u_id}, {song_id}, 1)
                    """
                cur.execute(query)
                conn.commit() 

            for tag in dislike["tags"]:
                query = f"""
                    INSERT into usr_gnr (u_id, gnr_name, isLike) values({u_id}, {tag}, 0)
                    """
                cur.execute(query)
                conn.commit()
            for song_id in dislike["songs"]:
                query = f"""
                    INSERT into usr_songs (u_id, song_id, isLike) values({u_id}, {song_id}, 0)
                    """
                cur.execute(query)
                conn.commit()

            output = inference(body, model, result_file_base.format(u_id))
            
            # output: true_like, maybe_like
            # laod_json: true_like
            return redirect('/playlist/songs')
        except (KeyError, JSONDecodeError, ValueError) as e:
            return JsonResponse({'success':False, 'ouput': e}, json_dumps_params={'ensure_ascii': True})

def show_songs(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()

    try:
        u_id = request.session['u_id']
        result = load_json(result_file_base.format(u_id))
        
        song_list = []

        for song in result['songs']:
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
        
        return JsonResponse({'success':True, 'tag_list': result['tags'], 'song_list': song_list}, json_dumps_params={'ensure_ascii': True})
    except (KeyError, JSONDecodeError, ValueError) as e:
        return JsonResponse({'success':False, 'ouput': e}, json_dumps_params={'ensure_ascii': True})