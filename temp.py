import flask
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import os
from keras.models import load_model
import base64
from retinaface import RetinaFace
import matplotlib.pyplot as plt


lmindex_lefteye = [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]
lmindex_righteye = [244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189]


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
        self.open=open

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

        righteye.open = False
        lefteye.open = False

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

            righteye.open = True
            lefteye.open = True

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
    faces = RetinaFace.extract_faces(img_path, align=True)
    detect_faces = RetinaFace.detect_faces(img_path)
    if detect_faces is None:
        return None

    data = []
    for faceNum in detect_faces.keys():
        identity = detect_faces[f'{faceNum}']
        facial_area = identity["facial_area"]
        eye_landmarks = [identity['landmarks']['right_eye'], identity['landmarks']['left_eye']]
        data.append((facial_area, *eye_landmarks, (facial_area[2] - facial_area[0], facial_area[3] - facial_area[1])))

    senddata = dict()
    senddata['people'] = len(data)
    for i in range(len(data)):
        re_x, re_y = data[i][1]
        le_x, le_y = data[i][2]
        size_x, size_y = data[i][3]
        senddata[f'face{i}'] = {
            'face': data[i][0],
            'righteye': {'pos': [int(re_x - size_x / 8), int(re_y - size_y / 16),
                                 int(re_x + size_x / 8), int(re_y + size_y / 16)], 'open': True},
            'lefteye': {'pos': [int(le_x - size_x / 8), int(le_y - size_y / 16),
                                int(le_x + size_x / 8), int(le_y + size_y / 16)], 'open': True}
        }
    return senddata


cv2.imwrite('res.png', make_sampleimg(['c.jpg', 'o.jpg']))
