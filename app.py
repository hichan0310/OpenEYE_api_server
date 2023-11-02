import random
import flask
from flask import Flask, request, send_file, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import load_model
import base64
from retinaface import RetinaFace
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import FloatTensor as tensor



# from classify_pytorch import classify_img


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# class_names = ['0 Opened', '1 Closed']
# model_path = "final_model.h5"
# model = load_model(f"{model_path}", compile=False)
#
# def classify_img(one_eye_img):  # input은 한 쪽 눈 이미지
#     img = cv2.resize(one_eye_img, (224, 224), interpolation=cv2.INTER_AREA)
#     img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
#
#     img = (img / 127.5) - 1
#
#     prediction = model.predict(img)
#     index = np.argmax(prediction)
#
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#     classified = class_name[2:]
#
#     return classified


# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     static_image_mode=True,
#     max_num_faces=100,
# )




class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 64, 48, 3 -> 32, 24, 16
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 32, 24, 16 -> 16, 12, 32
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=4)
        )
        # # 16, 12, 32 -> 8, 6, 64
        # self.layer3 = nn.Sequential(
        #     torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        # 8, 6, 64 -> 8*6*64=3072
        # 3072 -> 384 -> 48 -> 8 -> 1
        self.layer4 = nn.Sequential(
            nn.Linear(3072, 300),
            nn.ReLU(),
            nn.Linear(300, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x


model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

transform=transforms.ToTensor()
def classify_img(img):
    input_data=np.array(transform(cv2.resize(img, (64, 48))))
    input_data=np.array([input_data])
    input_data=tensor(input_data)
    result = model(input_data)
    return 0 if result[0][0]<1/2 else 1







app = Flask(__name__)


@app.route('/asdf', methods=['GET'])
def main():
    return flask.send_file('static/form.html')


@app.route('/overlay', methods=['POST', 'GET'])
def overlay():
    if request.method == 'POST':
        f = request.files['main']
        filepath_main = './save_image/overlay_background.jpg'
        f.save(filepath_main)
        f = request.files['overlay']
        filepath_overlay = './save_image/overlay.jpg'
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
        cv2.imwrite('./save_image/result_overlay.jpg', image_main)
        return send_file('./save_image/result_overlay.jpg', mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('./save_image/result_overlay.jpg', mimetype='image/jpg')


@app.route('/crop', methods=['POST', 'GET'])
def crop():
    if request.method == 'POST':
        f = request.files['image']
        # img_data = base64.b64decode(request.form['image'])
        filepath_main = './save_image/crop_image.jpg'
        f.save(filepath_main)
        # with open(filepath_main, 'wb') as f:
        #     f.write(img_data)

        image = cv2.imread(filepath_main)

        pos_topleft, pos_bottomright = request.form['topleft'], request.form['bottomright']
        pos_tlx, pos_tly = map(int, pos_topleft.split())
        pos_brx, pos_bry = map(int, pos_bottomright.split())

        image = image[pos_tly:pos_bry, pos_tlx:pos_brx]
        cv2.imwrite('./save_image/result_crop.jpg', image)
        return send_file('./save_image/result_crop.jpg', mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('./save_image/result_crop.jpg', mimetype='image/jpg')


# 눈 자르는 거 그냥 내가 만들었음

class EyePos:
    def __init__(self, size):
        self.max_x, self.max_y = 0, 0
        self.min_x, self.min_y, _ = size
        self.open = True

    def addpos(self, pos):
        self.max_x = max(self.max_x, pos[0])
        self.max_y = max(self.max_y, pos[1])
        self.min_x = min(self.min_x, pos[0])
        self.min_y = min(self.min_y, pos[1])

    def set(self, position, open):
        self.min_x, self.min_y, self.max_x, self.max_y = position
        self.open = open

    def size(self):
        return (self.max_x - self.min_x, self.max_y - self.min_y)

    def center(self):
        return (int((self.max_x + self.min_x) / 2), int((self.max_y + self.min_y) / 2))

    def move_center(self, new_center):
        center = self.center()
        movement = (new_center[0] - center[0], new_center[1] - center[1])
        self.min_x += movement[0]
        self.min_y += movement[1]
        self.max_x += movement[0]
        self.max_y += movement[1]


# lmindex_lefteye = [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]
# lmindex_righteye = [244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189]


def make_sampleimg(img_path_list):
    background_img = cv2.imread(img_path_list[0])
    imgRGB = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    img_size = imgRGB.shape
    result = get_face(img_path_list[0])
    bg_data_closed = []
    for i in range(result['people']):

        # pos_tlx, pos_tly = righteye.min_x, righteye.min_y
        # pos_brx, pos_bry = righteye.max_x, righteye.max_y
        # image = background[pos_tly:pos_bry, pos_tlx:pos_brx]
        # 모델 돌리기
        righteye = EyePos(img_size)
        lefteye = EyePos(img_size)

        righteye.set(result[f'face{i}']['righteye']['pos'], result[f'face{i}']['righteye']['open'])
        lefteye.set(result[f'face{i}']['lefteye']['pos'], result[f'face{i}']['lefteye']['open'])

        if not righteye.open:
            bg_data_closed.append(righteye)
        if not lefteye.open:
            bg_data_closed.append(lefteye)

    print(bg_data_closed)

    for img_path in img_path_list[1:]:
        img = cv2.imread(img_path)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = imgRGB.shape
        result = get_face(img_path)

        print(result['people'])
        for i in range(result['people']):
            righteye = EyePos(img_size)
            lefteye = EyePos(img_size)

            righteye.set(result[f'face{i}']['righteye']['pos'], result[f'face{i}']['righteye']['open'])
            lefteye.set(result[f'face{i}']['lefteye']['pos'], result[f'face{i}']['lefteye']['open'])

            for i in range(len(bg_data_closed)):
                openedeye = None
                if sameeye(bg_data_closed[i], lefteye) and lefteye.open:
                    openedeye = lefteye
                if sameeye(bg_data_closed[i], righteye) and righteye.open:
                    openedeye = righteye
                print(sameeye(bg_data_closed[i], righteye), sameeye(bg_data_closed[i], lefteye))
                if openedeye is not None:
                    tmp_eye = img[openedeye.min_y:openedeye.max_y, openedeye.min_x:openedeye.max_x]
                    openedeye.move_center(bg_data_closed[i].center())
                    for ii in range(openedeye.min_x, openedeye.max_x):
                        for jj in range(openedeye.min_y, openedeye.max_y):
                            background_img[jj][ii] = tmp_eye[jj - openedeye.min_y][ii - openedeye.min_x]
                    print(bg_data_closed[i])

    return background_img


def get_face(img_path):
    img = cv2.imread(img_path)
    detect_faces = RetinaFace.detect_faces(img_path)
    if detect_faces is None:
        return {'people': 0}
    try:
        data = []
        for faceNum in detect_faces.keys():
            identity = detect_faces[f'{faceNum}']
            facial_area = identity["facial_area"]
            eye_landmarks = [identity['landmarks']['right_eye'], identity['landmarks']['left_eye']]
            data.append(
                (facial_area, *eye_landmarks, (facial_area[2] - facial_area[0], facial_area[3] - facial_area[1])))

        senddata = dict()
        senddata['people'] = len(data)
        for i in range(len(data)):
            re_x, re_y = data[i][1]
            le_x, le_y = data[i][2]
            size_x, size_y = data[i][3]

            re_pos = [int(re_x - size_x / 8), int(re_y - size_y / 16),
                      int(re_x + size_x / 8), int(re_y + size_y / 16)]
            le_pos = [int(le_x - size_x / 8), int(le_y - size_y / 16),
                      int(le_x + size_x / 8), int(le_y + size_y / 16)]

            senddata[f'face{i}'] = {
                'face': list(map(int, data[i][0])),
                'righteye': {'pos': re_pos, 'open': classify_img(img[re_pos[1]:re_pos[3], re_pos[0]:re_pos[2]]) == 0},
                'lefteye': {'pos': le_pos, 'open': classify_img(img[le_pos[1]:le_pos[3], le_pos[0]:le_pos[2]]) == 0}
            }
        return senddata

    except:
        print("asdf")
        return {'people': 0}


@app.route('/eyepos', methods=['POST', 'GET'])
def eyepos():
    if request.method == 'POST':
        f = request.files['image']
        filepath_main = './save_image/eyepos.jpg'
        f.save(filepath_main)
        senddata = get_face(filepath_main)

        # f.save(filepath_main)
        # img = cv2.imread(filepath_main)
        # print(img.shape)
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_size = imgRGB.shape
        # results = face_mesh.process(imgRGB)
        # data = []
        # if results.multi_face_landmarks:
        #     for result in results.multi_face_landmarks:
        #         righteye, lefteye, face = EyePos(img_size), EyePos(img_size), EyePos(img_size)
        #         for lm_ind, lm in enumerate(result.landmark):
        #             if lm_ind in lmindex_lefteye:
        #                 lefteye.addpos((lm.x, lm.y))
        #             if lm_ind in lmindex_righteye:
        #                 righteye.addpos((lm.x, lm.y))
        #             face.addpos((lm.x, lm.y))
        #         # re_img = imgRGB[righteye.min_x:righteye.max_x, righteye.min_y:righteye.max_y]
        #         # le_img = imgRGB[lefteye.min_x:lefteye.max_x, lefteye.min_y:lefteye.max_y]
        #         # print(classify_img(re_img))
        #         # print(classify_img(le_img))
        #         # 왜인지는 모르겠지만 이걸 주석처리 안 하면 이미지 받는 부분에서 이상해짐
        #         data.append((righteye, lefteye, face))
        #
        # senddata = dict()
        # senddata['people'] = len(data)
        # for i in range(len(data)):
        #     senddata[f'face{i}'] = {
        #         'face': [int(data[i][2].min_x * img_size[0]),
        #                  int(data[i][2].min_y * img_size[1]),
        #                  int(data[i][2].max_x * img_size[0]),
        #                  int(data[i][2].max_y * img_size[1])],
        #         'righteye': {'pos': [int(data[i][0].min_x * img_size[0]),
        #                              int(data[i][0].min_y * img_size[1]),
        #                              int(data[i][0].max_x * img_size[0]),
        #                              int(data[i][0].max_y * img_size[1])], 'open': True},
        #         'lefteye': {'pos': [int(data[i][1].min_x * img_size[0]),
        #                             int(data[i][1].min_y * img_size[1]),
        #                             int(data[i][1].max_x * img_size[0]),
        #                             int(data[i][1].max_y * img_size[1])], 'open': True}
        #     }
        return jsonify(senddata)
    elif request.method == 'GET':
        return send_file('./save_image/eyepos.jpg', mimetype='image/jpg')


def sameeye(eye1: EyePos, eye2: EyePos):
    mx1, my1, Mx1, My1 = eye1.min_x, eye1.min_y, eye1.max_x, eye1.max_y
    mx2, my2, Mx2, My2 = eye2.min_x, eye2.min_y, eye2.max_x, eye2.max_y
    if (mx1 <= mx2 <= Mx1 and my1 <= my2 <= My1) \
            or (mx1 <= mx2 <= Mx1 and my1 <= My2 <= My1) \
            or (mx1 <= Mx2 <= Mx1 and my1 <= my2 <= My1) \
            or (mx1 <= Mx2 <= Mx1 and my1 <= My2 <= My1):
        return True
    elif (mx2 <= mx1 <= Mx2 and my2 <= my1 <= My2) \
            or (mx2 <= mx1 <= Mx2 and my2 <= My1 <= My2) \
            or (mx2 <= Mx1 <= Mx2 and my2 <= my1 <= My2) \
            or (mx2 <= Mx1 <= Mx2 and my2 <= My1 <= My2):
        return True
    else:
        return False


def make_sampleimg(img_path_list):
    background_img = cv2.imread(img_path_list[0])
    imgRGB = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    img_size = imgRGB.shape
    result = get_face(img_path_list[0])
    bg_data_closed = []
    for i in range(result['people']):

        # pos_tlx, pos_tly = righteye.min_x, righteye.min_y
        # pos_brx, pos_bry = righteye.max_x, righteye.max_y
        # image = background[pos_tly:pos_bry, pos_tlx:pos_brx]
        # 모델 돌리기
        righteye = EyePos(img_size)
        lefteye = EyePos(img_size)

        righteye.set(result[f'face{i}']['righteye']['pos'], result[f'face{i}']['righteye']['open'])
        lefteye.set(result[f'face{i}']['lefteye']['pos'], result[f'face{i}']['lefteye']['open'])

        if not righteye.open:
            bg_data_closed.append(righteye)
        if not lefteye.open:
            bg_data_closed.append(lefteye)

    print(bg_data_closed)

    for img_path in img_path_list[1:]:
        img = cv2.imread(img_path)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = imgRGB.shape
        result = get_face(img_path)

        print(result['people'])
        for i in range(result['people']):
            righteye = EyePos(img_size)
            lefteye = EyePos(img_size)

            righteye.set(result[f'face{i}']['righteye']['pos'], result[f'face{i}']['righteye']['open'])
            lefteye.set(result[f'face{i}']['lefteye']['pos'], result[f'face{i}']['lefteye']['open'])

            for ind in range(len(bg_data_closed)):
                i = len(bg_data_closed) - ind - 1
                openedeye = None
                if sameeye(bg_data_closed[i], lefteye) and lefteye.open:
                    openedeye = lefteye
                if sameeye(bg_data_closed[i], righteye) and righteye.open:
                    openedeye = righteye
                print(sameeye(bg_data_closed[i], righteye), sameeye(bg_data_closed[i], lefteye))
                if openedeye is not None:
                    tmp_eye = img[openedeye.min_y:openedeye.max_y, openedeye.min_x:openedeye.max_x]
                    openedeye.move_center(bg_data_closed[i].center())
                    for ii in range(openedeye.min_x, openedeye.max_x):
                        for jj in range(openedeye.min_y, openedeye.max_y):
                            background_img[jj][ii] = tmp_eye[jj - openedeye.min_y][ii - openedeye.min_x]
                    print(bg_data_closed[i])
                    del (bg_data_closed[i])

    return background_img


@app.route('/sampleimg', methods=['POST', 'GET'])
def sampleimg():
    if request.method == 'POST':
        f = request.files['image1']
        filepath_main = './save_image/sample1.jpg'
        f.save(filepath_main)
        f = request.files['image2']
        filepath_main = './save_image/sample2.jpg'
        f.save(filepath_main)
        f = request.files['image3']
        filepath_main = './save_image/sample3.jpg'
        f.save(filepath_main)

        result_img = make_sampleimg(
            ['./save_image/sample1.jpg', './save_image/sample2.jpg', './save_image/sample3.jpg'])
        cv2.imwrite('./save_image/sample_img.jpg', result_img)
        return send_file('./save_image/sample_img.jpg', mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('./save_image/sample_img.jpg', mimetype='image/jpg')


@app.route('/rotate', methods=['POST', 'GET'])
def rotate():
    if request.method == 'POST':
        f = request.files['image']
        filepath = './save_image/isopen.jpg'
        f.save(filepath)
        img = cv2.imread(filepath)
        angle = request.form['angle']
        img = cv2.rotate(img, int(angle))
        cv2.imwrite('./save_image/rotate.jpg', img)
        return send_file('./save_image/rotate.jpg', mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('./save_image/rotate.jpg', mimetype='image/jpg')


# 눈 떴으면 True, 눈 감았으면 False
@app.route('/isopen', methods=['POST', 'GET'])
def isopen():
    if request.method == 'POST':
        f = request.files['image']
        filepath = './save_image/isopen.jpg'
        f.save(filepath)
        img = cv2.imread(filepath)
        return str(classify_img(img) == 0)
    if request.method == 'GET':
        return str(None)



@app.route('/changeeye', methods=['POST', 'GET'])
def changeeye():
    if request.method == 'POST':
        f = request.files['img']
        filepath = './save_image/changeeye_img.jpg'
        f.save(filepath)
        bg_img=cv2.imread(filepath)

        f = request.files['eye']
        filepath = './save_image/changeeye_eye.jpg'
        f.save(filepath)
        eye_img=cv2.imread(filepath)

        h, w, c = eye_img.shape

        pos_center = list(map(int, request.form['center'].split()))



        for i in range(w):
            for j in range(h):
                bg_img[pos_center[1] - h//2 + j][pos_center[0] - w//2 + i] = eye_img[j][i]
        cv2.imwrite('./save_image/changeeye_result.jpg', bg_img)

        return send_file('./save_image/changeeye_result.jpg', mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('./save_image/changeeye_result.jpg', mimetype='image/jpg')







# 정윤이가 만든 거 api로 적용하는 것까지만 하면 되려나


if __name__ == '__main__':
    app.run()
