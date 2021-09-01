from django.urls import path, include
from django.conf import settings

from . import views

app_name = 'Playlist'

urlpatterns = [
    path('', views.index, name='index'),
    path('detail', views.detail, name='detail'),
    path('recommend', views.show_inference, name='recommend')
]