import flask
from flask import Flask, request, send_file, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import load_model
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
class_names = ['0 Opened', '1 Closed']
model_path = "final_model.h5"
model = load_model(f"{model_path}", compile=False)


def classify_img(one_eye_img):  # input은 한 쪽 눈 이미지
    img = cv2.resize(one_eye_img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)

    img = (img / 127.5) - 1

    prediction = model.predict(img)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]
    classified = class_name[2:]

    return classified


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=100,
)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    return flask.send_file('static/form.html')


@app.route('/overlay', methods=['POST', 'GET'])
def overlay():
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
        cv2.imwrite('./save_image/result_overlay.png', image_main)
        return send_file('./save_image/result_overlay.png', mimetype='image/png')
    elif request.method == 'GET':
        return send_file('./save_image/result_overlay.png', mimetype='image/png')


@app.route('/crop', methods=['POST', 'GET'])
def crop():
    if request.method == 'POST':
        # f = request.files['image']
        img_data = base64.b64decode(request.form['image'])
        filepath_main = './save_image/crop_image.png'
        # f.save(filepath_main)
        with open(filepath_main, 'wb') as f:
            f.write(img_data)

        image = cv2.imread(filepath_main)

        pos_topleft, pos_bottomright = request.form['topleft'], request.form['bottomright']
        pos_tlx, pos_tly = map(int, pos_topleft.split())
        pos_brx, pos_bry = map(int, pos_bottomright.split())

        image = image[pos_tly:pos_bry, pos_tlx:pos_brx]
        cv2.imwrite('./save_image/result_crop.png', image)
        return send_file('./save_image/result_crop.png', mimetype='image/png')
    elif request.method == 'GET':
        return send_file('./save_image/result_crop.png', mimetype='image/png')


# 눈 자르는 거 그냥 내가 만들었음

class EyePos:
    def __init__(self, size):
        self.max_x, self.max_y = 0, 0
        self.min_x, self.min_y, _ = size

    def addpos(self, pos):
        self.max_x = max(self.max_x, pos[0])
        self.max_y = max(self.max_y, pos[1])
        self.min_x = min(self.min_x, pos[0])
        self.min_y = min(self.min_y, pos[1])


lmindex_lefteye = [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]
lmindex_righteye = [244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189]


@app.route('/eyepos', methods=['POST', 'GET'])
def eyepos():
    if request.method == 'POST':
        f = request.files['image']
        filepath_main = './save_image/eyepos.png'
        f.save(filepath_main)
        img = cv2.imread(filepath_main)
        print(img.shape)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = imgRGB.shape
        results = face_mesh.process(imgRGB)
        data = []
        if results.multi_face_landmarks:
            for result in results.multi_face_landmarks:
                righteye, lefteye, face = EyePos(img_size), EyePos(img_size), EyePos(img_size)
                for lm_ind, lm in enumerate(result.landmark):
                    if lm_ind in lmindex_lefteye:
                        lefteye.addpos((lm.x, lm.y))
                    if lm_ind in lmindex_righteye:
                        righteye.addpos((lm.x, lm.y))
                    face.addpos((lm.x, lm.y))
                # re_img = imgRGB[righteye.min_x:righteye.max_x, righteye.min_y:righteye.max_y]
                # le_img = imgRGB[lefteye.min_x:lefteye.max_x, lefteye.min_y:lefteye.max_y]
                # print(classify_img(re_img))
                # print(classify_img(le_img))
                # 왜인지는 모르겠지만 이걸 주석처리 안 하면 이미지 받는 부분에서 이상해짐
                data.append((righteye, lefteye, face))

        senddata = dict()
        senddata['people'] = len(data)
        for i in range(len(data)):
            senddata[f'face{i}'] = {
                'face': [int(data[i][2].min_x * img_size[0]),
                         int(data[i][2].min_y * img_size[1]),
                         int(data[i][2].max_x * img_size[0]),
                         int(data[i][2].max_y * img_size[1])],
                'righteye': [int(data[i][2].min_x * img_size[0]),
                             int(data[i][0].min_y * img_size[1]),
                             int(data[i][0].max_x * img_size[0]),
                             int(data[i][0].max_y * img_size[1])],
                'lefteye': [int(data[i][1].min_x * img_size[0]),
                            int(data[i][1].min_y * img_size[1]),
                            int(data[i][1].max_x * img_size[0]),
                            int(data[i][1].max_y * img_size[1])]
            }
        return jsonify(senddata)
    elif request.method == 'GET':
        return send_file('./save_image/eyepos.png', mimetype='image/png')


@app.route('/isopen', methods=['POST', 'GET'])
def isopen():
    if request.method == 'POST':
        f = request.files['image']
        filepath = './save_image/isopen.png'
        f.save(filepath)
        img = cv2.imread(filepath)
        return str(classify_img(img))


# 정윤이가 만든 거 api로 적용하는 것까지만 하면 되려나


if __name__ == '__main__':
    app.run()
