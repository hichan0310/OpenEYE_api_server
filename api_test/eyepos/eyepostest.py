import requests

#눈 찾기 api 사용법
# file=open('geteye.png', 'rb')
#
# upload = {'image': file}
#
# res = requests.post(' http://127.0.0.1:5000/eyepos', files = upload)
# print(res.json())

file=open('geteye.png', 'rb')
upload = {'image': file}
data={'topleft':'116 190', 'bottomright':'147 208'}

requests.post(' http://127.0.0.1:5000/crop', files = upload, data=data)
