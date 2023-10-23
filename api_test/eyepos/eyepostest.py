import time

import requests

#눈 찾기 api 사용법
file=open('test.png', 'rb')
print(file)
upload = {'image': file}
analyse_result:dict
while True:
    res = requests.post('http://127.0.0.1:5000/eyepos', files = upload, timeout=10)
    if res.status_code==200:
        analyse_result=res.json()
        break
    time.sleep(10)

for i in range(analyse_result['people']):
    print(' '.join(list(map(str, analyse_result[f'face{i}']['lefteye']))), '\n', ' '.join(list(map(str, analyse_result[f'face{i}']['righteye']))))
print(analyse_result)

file=open('test.png', 'rb')
upload = {'image': file}
data={'topleft':'561 2038', 'bottomright':'781 2171'}
while True:
    res=requests.post('http://127.0.0.1:5000/crop', files = upload, data=data)
    if res.status_code==200:
        break

for i in range(8):
    file=open(f'{i}.png', 'rb')
    upload = {'image': file}
    res=requests.post('http://127.0.0.1:5000/isopen', files=upload)
    print(res.text)


print('fin')

