import time

import requests

#눈 찾기 api 사용법
file=open('geteye.png', 'rb')
print(file)
upload = {'image': file}
analyse_result:dict
while True:
    res = requests.post(' http://127.0.0.1:5000/eyepos', files = upload, timeout=10)
    if res.status_code==200:
        analyse_result=res.json()
        break
    time.sleep(10)

for i in range(analyse_result['people']):
    print(analyse_result[f'face{i}'])
print(analyse_result)

file=open('geteye.png', 'rb')
upload = {'image': file}
data={'topleft':'65 190', 'bottomright':'147 208'}
while True:
    res=requests.post(' http://127.0.0.1:5000/crop', files = upload, data=data)
    if res.status_code==200:
        break
print('fin')
