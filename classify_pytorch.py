import numpy as np
import torch
from model_class import Model
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import FloatTensor as tensor

# model_path = 'model_fold_20.pth'
# model = Model()
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# def classify_img(image): # input은 한 쪽 눈 이미지
#     # 1 : closed, 0 : opened
#     image = image.astype(np.float32) / 255.0
#     image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
#
#     with torch.no_grad():
#         output = model(image)
#
#     predicted_class = output.argmax(dim=1).item()
#
#     return predicted_class
#
# # for i in range(8):
# #     print(classify_img(cv2.imread(f'./api_test/eyepos/{i}.png')))  # 0







# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 64, 48, 3 -> 32, 24, 16
#         self.layer1 = nn.Sequential(
#             torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # 32, 24, 16 -> 16, 12, 32
#         self.layer2 = nn.Sequential(
#             torch.nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=4, stride=4)
#         )
#         # # 16, 12, 32 -> 8, 6, 64
#         # self.layer3 = nn.Sequential(
#         #     torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#         #     torch.nn.ReLU(),
#         #     torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         # )
#
#         # 8, 6, 64 -> 8*6*64=3072
#         # 3072 -> 384 -> 48 -> 8 -> 1
#         self.layer4 = nn.Sequential(
#             nn.Linear(3072, 300),
#             nn.ReLU(),
#             nn.Linear(300, 20),
#             nn.ReLU(),
#             nn.Linear(20, 1),
#             nn.Sigmoid()
#         )
#
#         # self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         # x = self.layer3(x)
#         x = x.view(x.size(0), -1)
#         x = self.layer4(x)
#         return x
#
#
# model = CNN()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()
#
# transform=transforms.ToTensor()
# def classify_img(img):
#     input_data=np.array(transform(cv2.resize(img, (64, 48))))
#     input_data=np.array([input_data])
#     input_data=tensor(input_data)
#     result = model(input_data)
#     return 0 if result[0][0]<1/2 else 1
#
# print(classify_img(cv2.imread('./api_test/eyepos/0.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/1.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/2.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/3.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/4.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/5.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/6.png')))
# print(classify_img(cv2.imread('./api_test/eyepos/7.png')))
# print()
# print(classify_img(cv2.imread('0.4364540599071556_0.png')))
# print(classify_img(cv2.imread('0.5095575715455499_0.png')))
# print(classify_img(cv2.imread('0.5800064141197784_0.png')))
# print(classify_img(cv2.imread('0.8024058370529631_0.png')))



import random
import flask
from flask import Flask, request, send_file, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from keras.models import load_model
import base64
from retinaface import RetinaFace
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import FloatTensor as tensor

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


lmindex_lefteye = [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]
lmindex_righteye = [244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189]


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=1,
)


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
            face=list(map(int, data[i][0]))
            face_img=img[face[0]:face[2], face[1]:face[3]]
            face_size=(face[2]-face[0], face[3]-face[1])
            results=face_mesh.process(face_img)
            re_pos=[]
            le_pos=[]
            print(1)
            if (type(results.multi_face_landmarks) is list):
                re, le = EyePos(face_size), EyePos(face_size)
                for result in results.multi_face_landmarks:
                    for id, lm in enumerate(result.landmark):
                        if id in lmindex_righteye:
                            re.addpos((int(lm.x*face_size[0]), int(lm.y*face_size[1])))
                        if id in lmindex_lefteye:
                            le.addpos((int(lm.x*face_size[0]), int(lm.y*face_size[1])))
                re_pos = [face[0] + re.min_x, face[0] + re.max_x, face[1] + re.min_y, face[1] + re.max_y]
                le_pos = [face[0] + le.min_x, face[0] + le.max_x, face[1] + le.min_y, face[1] + le.max_y]
            else:
                print(2)
                re_x, re_y = data[i][1]
                le_x, le_y = data[i][2]
                size_x, size_y = data[i][3]



                re_pos = [int(re_x - size_x / 8), int(re_y - size_y / 16),
                          int(re_x + size_x / 8), int(re_y + size_y / 16)]
                le_pos = [int(le_x - size_x / 8), int(le_y - size_y / 16),
                          int(le_x + size_x / 8), int(le_y + size_y / 16)]

            senddata[f'face{i}'] = {
                'face': list(map(int, data[i][0])),
                'righteye': {'pos': re_pos, 'open': True},
                'lefteye': {'pos': le_pos, 'open': True}
            }
        return senddata

    except:
        print("asdf")
        return {'people': 0}


print('asdf')
print(get_face('o.jpg'))






