from rest_framework import serializers
from .models import Url

class UrlSerializer(serializers.ModelSerializer):
    class Meta:
        model = Url
        fields = ['uuid', 'title', 'url']
        read_only_fields = ['uuid']
