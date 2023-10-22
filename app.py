from flask import Flask, request, send_file, jsonify
import cv2
import mediapipe as mp
import time
import numpy as np
from torch import FloatTensor as tensor
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=1,
)

app = Flask(__name__)


@app.route('/')
def nothing():
    return 'api_test 폴더에 api 사용 예시 파이썬 코드가 나와있습니다'


@app.route('/overlay', methods=['POST', 'GET'])
def overlay():
    print(request.method)
    if request.method == 'POST':
        f = request.files['main']
        filepath_main = './save_image/overlay_background.png'
        f.save(filepath_main)
        f = request.files['overlay']
        filepath_overlay = './save_image/overlay.png'
        f.save(filepath_overlay)

        image_main = cv2.imread(filepath_main)
        image_overlay = cv2.imread(filepath_overlay)

        pos_topleft, pos_bottomright = request.form['topleft'], request.form['bottomright']
        pos_topleft = tuple(map(int, pos_topleft.split()))
        pos_bottomright = tuple(map(int, pos_bottomright.split()))
        size = (pos_bottomright[0] - pos_topleft[0], pos_bottomright[1] - pos_topleft[1])

        image_overlay = cv2.resize(image_overlay, size)
        for i in range(size[0]):
            for j in range(size[1]):
                image_main[pos_topleft[1] + j][pos_topleft[0] + i] = image_overlay[j][i]
        cv2.imwrite('./save_image/result.png', image_main)
        return send_file('./save_image/result.png', mimetype='image/png')
    elif request.method == 'GET':
        return send_file('./save_image/result.png', mimetype='image/png')


@app.route('/crop', methods=['POST', 'GET'])
def crop():
    print(request.method)
    if request.method == 'POST':
        f = request.files['main']
        filepath_main = './save_image/crop_image.png'
        f.save(filepath_main)

        image = cv2.imread(filepath_main)

        pos_topleft, pos_bottomright = request.form['topleft'], request.form['bottomright']
        pos_tlx, pos_tly = map(int, pos_topleft.split())
        pos_brx, pos_bry = map(int, pos_bottomright.split())

        image = image[pos_tly:pos_bry, pos_tlx:pos_brx]
        cv2.imwrite('./save_image/result.png', image)
        return send_file('./save_image/result.png', mimetype='image/png')
    elif request.method == 'GET':
        return send_file('./save_image/result.png', mimetype='image/png')




# 눈 자르는 거 그냥 내가 만들었음

class EyePos:
    def __init__(self, img_size):
        self.max_x, self.max_y=0, 0
        self.min_x, self.min_y=img_size

    def addpos(self, pos):
        self.max_x=max(self.max_x, pos[0])
        self.max_y=max(self.max_y, pos[1])
        self.min_x=min(self.min_x, pos[0])
        self.min_y=min(self.min_y, pos[1])


lmindex_lefteye=[464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]
lmindex_righteye=[244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189]
@app.route('/eyepos', methods=['POST', 'GET'])
def eyepos():
    f = request.files['main']
    filepath_main = './save_image/eyepos.png'
    f.save(filepath_main)
    img = cv2.imread(filepath_main)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size=imgRGB.size
    results = face_mesh.process(imgRGB)
    data=[]
    if results.multi_face_landmarks:
        for result in results.multi_face_landmarks:
            righteye, lefteye=EyePos(img_size), EyePos(img_size)
            m_x, m_y, M_x, M_y = 0, 0, 0, 0
            for lm_ind, lm in enumerate(result.landmark):
                if lm_ind in lmindex_lefteye:
                    lefteye.addpos((lm.x, lm.y))
                if lm_ind in lmindex_righteye:
                    righteye.addpos((lm.x, lm.y))
            data.append((righteye, lefteye))
    senddata=dict()
    senddata['people']=len(data)
    for i in range(len(data)):
        senddata[f'face{i}']={
            'righteye':{'x1':data[i][0].min_x, 'y1':data[i][0].min_y, 'x2':data[i][0].max_x, 'y2':data[i][0].max_y}
        }
    return jsonify(senddata)



# 정윤이가 만든 거 api로 적용하는 것까지만 하면 되려나


if __name__ == '__main__':
    app.run()
