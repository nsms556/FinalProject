from sqlite3.dbapi2 import IntegrityError, connect
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
import json
import sqlite3

@method_decorator(csrf_exempt, name='dispatch')
def index(request):
    
    # 마이페이지 (사용자 호/불호 장르, 태그, 가수 조회)
    if request.method == 'GET':
        try:
            u_id = request.session['u_id']
            conn = sqlite3.connect('data.db')
            cur = conn.cursor()
            query = f"""
                select a.id, a.song_name, a.artist_name_basket, a.issue_date
                from song_meta a, usr_song b
                where b.u_id = {u_id} and a.id = b.song_id and b.isLike=0
            """
            cur.execute(query)
            dislike_songs = []
            for row in cur.fetchall():
                content = {
                    'song_id': row[0],
                    'song_name': row[1],
                    'artist_name': row[2],
                    'issue_date': row[3]
                }
                dislike_songs.append(content)

            query = f"""
                select a.id, a.song_name, a.artist_name_basket, a.issue_date
                from song_meta a, usr_song b
                where b.u_id = {u_id} and a.id = b.song_id and b.isLike=1
            """
            cur.execute(query)
            rows = cur.fetchall()
            like_songs = []
            for row in rows:
                content = {
                    'song_id': row[0],
                    'song_name': row[1],
                    'artist_name': row[2],
                    'issue_date': row[3]
                }
                like_songs.append(content)
            
            return JsonResponse({"success": True, "like": like_songs, "dislike": dislike_songs}, json_dumps_params={'ensure_ascii': True}) 

        except KeyError as e:
            # 홈으로 리다이랙트
            return JsonResponse({"success": False, "error": str(type(e))})

    elif request.method == 'POST':
        try:
            u_id = request.session['u_id']
            body = json.loads(request.body.decode('utf-8'))
            conn = sqlite3.connect('data.db')
            cur = conn.cursor()
            query = f"""
                delete from usr_song where u_id = {u_id} and song_id = {body["song_id"]}
            """
            cur.execute(query)
            conn.commit()
            return JsonResponse({"success": True}, json_dumps_params={'ensure_ascii': True}) 

        except KeyError as e:
            # 홈으로 리다이랙트
            return JsonResponse({"success": False, "error": str(type(e))})

    
#region Auth
@method_decorator(csrf_exempt, name='dispatch')
def register(request):
    # 회원가입 화면
    if request.method == 'GET':
        return JsonResponse({"success":True}, json_dumps_params={'ensure_ascii': True}) 

    elif request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            username = body['username'] # 로그인 id
            password = body['password']

            # Create user and save to the database
            user = User.objects.create_user(username=username, password=password)
            user.save()
            # return redirect('index')
            return JsonResponse({'success':True, 'name':username}, json_dumps_params={'ensure_ascii': True})
        
        except IntegrityError as e:
            # 같은 유저 정보로 또 회원가입할 시 오류 예외처리 필요
            return JsonResponse({'success':False, 'status': '오류 발생', 'error': e}, json_dumps_params={'ensure_ascii': True})

@method_decorator(csrf_exempt, name='dispatch')
def signin(request):
    # 로그인 화면
    if request.method == 'GET':
        return JsonResponse({"success":True}, json_dumps_params={'ensure_ascii': True}) 

    elif request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            username = body['username']
            password = body['password']
            user = authenticate(request, username = username, password = password)

            if user is not None:
                request.session['u_id'] = user.pk
                return JsonResponse({'success':True, 'status': '로그인 성공'}, json_dumps_params={'ensure_ascii': True})
            else:
                return JsonResponse({'success':False, 'status': '로그인 실패'}, json_dumps_params={'ensure_ascii': True})
        except IntegrityError as e:
            # 같은 유저 정보로 또 회원가입할 시 오류 예외처리 필요
            return JsonResponse({'success':False, 'status': '오류 발생', 'error': e}, json_dumps_params={'ensure_ascii': True})

def signout(request):
    # 이미 로그인되있는 상태인지 예외처리 필요
    u_id = request.session['u_id']
    logout(request)
    return JsonResponse({'success':True, 'status': f'{u_id} 로그아웃 하였습니다'}, json_dumps_params={'ensure_ascii': True})
#endregion

# 장르 호/불호 입력
@method_decorator(csrf_exempt, name='dispatch')
def SelectGnr(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    # 선택할 장르 보여주기
    if request.method == 'GET':
        try:
            gn_gnr = []
            query = f"""
                select *
                from genre_meta
                where gnr_code like '%00'
            """
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                content = {
                    'gnr_id': row[0],
                    'gnr_code': row[1],
                    'gnr_name': row[2]
                }
                gn_gnr.append(content)

            return JsonResponse({"success":True, "gn_gnr": gn_gnr}, json_dumps_params={'ensure_ascii': True}) 
        except KeyError as e:
            return JsonResponse({"success": False, "error": str(type(e))})

    # 장르 선택
    elif request.method == 'POST':
        try:
            u_id = request.session['u_id']
            body = json.loads(request.body.decode('utf-8'))

            # 중복 튜플 insert 방지 필요
            cur.execute(f"""
                insert into user_gnr values ({u_id}, {body["gnr_id"]}, '{body["gnr_code"]}', {body["isLike"]})
            """)
            conn.commit()
            return JsonResponse({'success':True, 'output': 'save genre preference'}, json_dumps_params={'ensure_ascii': True})

        except KeyError as e:
            return JsonResponse({'success':False, 'error': str(type(e))}, json_dumps_params={'ensure_ascii': True})

# usr_song
@method_decorator(csrf_exempt, name='dispatch')
def SelectSong(request):
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    # 호/불호 노래 입력 (이부분 추천쿼리 돌리기 전에 한번에 받을지)
    if request.method == 'POST':
        try:
            u_id = request.session['u_id']
            body = json.loads(request.body.decode('utf-8'))
            query = f"""
                INSERT into usr_song values({u_id}, {body["song_id"]}, {body["isLike"]})
            """
            cur.execute(query)
            conn.commit()
            return JsonResponse({"success":True, "song": body["song_id"]}, json_dumps_params={'ensure_ascii': True}) 
        except KeyError as e:
            return JsonResponse({"success": False, "error": str(type(e))})