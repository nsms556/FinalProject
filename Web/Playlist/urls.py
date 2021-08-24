from django.urls import path, include
from django.conf import settings

from . import views

app_name = 'Playlist'

urlpatterns = [
    path('', views.index, name='index'),
]