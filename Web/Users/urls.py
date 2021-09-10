from os import name
from django.urls import path, include
from django.conf import settings

from . import views

app_name = 'Users'

urlpatterns = [
    path('', views.index, name='index'),
    path('register', views.register, name='register'),
    path('login', views.signin, name='login'),
    path('logout', views.signout, name='logout'),

    path('select_gnr', views.SelectGnr, name='select_gnr'),
    path('select_song', views.SelectSong, name='select_song')
]