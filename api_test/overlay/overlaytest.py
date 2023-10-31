import requests

# 오버레이 api 사용방법
file = open('D:\OpenEYE_api_server\o.jpg', 'rb')
overlay=open('saki.png', 'rb')

upload = {'image': file}
data={'topleft':'1926 1029',
      'bottomright':'2682 2444'}

res = requests.post('https://openeye.ziho.kr//crop', files = upload, data=data)
