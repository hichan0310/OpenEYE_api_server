import requests

#눈 찾기 api 사용법
file=open('test.png', 'rb')
upload = {'image': file}
analyse_result:dict
while True:
    res = requests.post(' http://127.0.0.1:5000/eyepos', files = upload)
    if res.status_code==200:
        analyse_result=res.json()
        print(analyse_result['people'])
        break

for i in range(analyse_result['people']):
    print(analyse_result[f'face{i}'])

file=open('test.png', 'rb')
upload = {'image': file}
data={'topleft':'1949 533', 'bottomright':'2455 689'}
while True:
    res=requests.post(' http://127.0.0.1:5000/crop', files = upload, data=data)
    if res.status_code==200:
        break
print('fin')
