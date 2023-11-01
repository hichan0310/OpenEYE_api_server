import cv2, os
from retinaface import RetinaFace

class Face:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.faces = RetinaFace.extract_faces(img_path=img_path, align=False, threshold=0.5)
        self.detect_faces = RetinaFace.detect_faces(img_path)
        self.sorted_people = [] #사진 왼쪽부터 정렬
    def detect(self, show_results = True):
        if self.detect_faces is None:
            return None
        else:
            tmp = []
            for faceNum in self.detect_faces.keys():
                identity = self.detect_faces[f'{faceNum}']
                facial_area = identity["facial_area"]
                facial_img = self.img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                tmp.append([faceNum, facial_img, facial_area[0]])


                if show_results == True:
                    cv2.rectangle(self.img, (facial_area[0], facial_area[1])
                                  , (facial_area[2], facial_area[3]), (255, 255, 255), 1)
                    cv2.putText(self.img, f'{faceNum}', (facial_area[0], facial_area[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            tmp = sorted(tmp, key=lambda x: x[2])
            self.sorted_people = list(map(lambda x: x[:2], tmp))
            if show_results== True:
                return self.img

        return self.sorted_people #[faceNum, facial_img] 탐지된 번호, 얼굴 이미지 순으로 담은 리스트

    def num_people(self):
        return len(self.sorted_people)
    def save_img(self, inputnum, index, faceNum, facial_img):
        path = f'facial_image\\input_{inputnum}\\image_{index}\\{faceNum}'
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(path + '\\face.jpg', facial_img)








