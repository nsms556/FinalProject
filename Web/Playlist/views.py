from django.shortcuts import render
import os
import json
from django.conf import settings

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
