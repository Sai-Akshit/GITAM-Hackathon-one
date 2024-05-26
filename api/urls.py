from django.urls import path
from . import views

urlpatterns = [
    path('list-url/', views.urlList, name='list-url'),
    path('add-url/', views.getUrl, name='add-url'),
]