from sqlite3.dbapi2 import IntegrityError
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
import json

def index(request):
    return JsonResponse({'success':True, "output":"users"}, json_dumps_params={'ensure_ascii': True})

@method_decorator(csrf_exempt, name='dispatch')
def register(request):
    # 회원가입 화면
    if request.method == 'GET':
        return JsonResponse({"success":True}, json_dumps_params={'ensure_ascii': True}) 

    elif request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            username = body['username']
            email = body['email']
            password = body['password']

            # Create user and save to the database
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            # return redirect('index')
            return JsonResponse({'success':True, 'email': email, 'name':username}, json_dumps_params={'ensure_ascii': True})
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
                request.session['user_id'] = user.pk
                # print(request.session.session_key, request.session['user_id'])
                return JsonResponse({'success':True, 'status': '로그인 성공'}, json_dumps_params={'ensure_ascii': True})
            else:
                return JsonResponse({'success':False, 'status': '로그인 실패'}, json_dumps_params={'ensure_ascii': True})
        except IntegrityError as e:
            # 같은 유저 정보로 또 회원가입할 시 오류 예외처리 필요
            return JsonResponse({'success':False, 'status': '오류 발생', 'error': e}, json_dumps_params={'ensure_ascii': True})

def signout(request):
    logout(request)
    return JsonResponse({'success':True, 'status': '로그아웃 하였습니다'}, json_dumps_params={'ensure_ascii': True})
