import requests


# 오버레이 api 사용방법
file = open('nounderstand.png', 'rb')
overlay=open('saki.png', 'rb')

upload = {'main': file,
          'overlay': overlay}
data={'topleft':'1100 500',
      'bottomright':'1900 1000'}

res = requests.post(' http://127.0.0.1:5000/overlay', files = upload, data=data)
