import uuid
from django.db import models


class Url(models.Model):
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)    # better if made primary key
    title = models.CharField(max_length=255)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)


class Chat(models.Model):
    url = models.ForeignKey(Url, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    role = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
