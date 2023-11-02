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

print(classify_img(cv2.imread('./api_test/eyepos/0.png')))
print(classify_img(cv2.imread('./api_test/eyepos/1.png')))
print(classify_img(cv2.imread('./api_test/eyepos/2.png')))
print(classify_img(cv2.imread('./api_test/eyepos/3.png')))
print(classify_img(cv2.imread('./api_test/eyepos/4.png')))
print(classify_img(cv2.imread('./api_test/eyepos/5.png')))
print(classify_img(cv2.imread('./api_test/eyepos/6.png')))
print(classify_img(cv2.imread('./api_test/eyepos/7.png')))
print()
print(classify_img(cv2.imread('0.4364540599071556_0.png')))
print(classify_img(cv2.imread('0.5095575715455499_0.png')))
print(classify_img(cv2.imread('0.5800064141197784_0.png')))
print(classify_img(cv2.imread('0.8024058370529631_0.png')))

