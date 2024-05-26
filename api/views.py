from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import Url, Chat
from .serializers import UrlSerializer


@api_view(['GET'])
def urlList(request):
    urls = Url.objects.all()
    serializer = UrlSerializer(urls, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def getUrl(request):
    serializer = UrlSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()

        url = serializer.data['url']

        res = 'completed'   # fetch this result from the embedding function
        
        # should modify the below code

        if res == 'completed':
            return Response(serializer.data, status=status.HTTP_200_OK)
        elif res == 'error':
            return Response({'message': 'Error'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def getUserResponse(request, url_uuid):
    urlObj = get_object_or_404(Url, uuid=url_uuid)

    user_input = request.data['user_input']
    
    if not user_input:
        return Response({'message': 'Please provide user input'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        Chat.objects.create(url=urlObj, message=user_input, role='user')

        system_output = 'output'
        
        Chat.objects.create(url=urlObj, message=system_output, role='system')

        return Response({'system_output': system_output}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': 'An error occurred while processing the request'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
