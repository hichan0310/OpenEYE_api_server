import cv2
import mediapipe as mp

class Eye:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils

    def cropEye(self, img): #직사각형으로 눈 영역 자르기
        image_width = img.shape[1]
        image_height = img.shape[0]
        l_eye = None
        r_eye = None
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            for single_face_landmarks in results.multi_face_landmarks:
                l_eye_idx = (46, 128)  # 직사각형 왼쪽 위, 오른쪽 아래 좌표
                r_eye_idx = (285, 340)

                l_eye_coord_1 = single_face_landmarks.landmark[l_eye_idx[0]]
                l_eye_coord_2 = single_face_landmarks.landmark[l_eye_idx[1]]

                r_eye_coord_1 = single_face_landmarks.landmark[r_eye_idx[0]]
                r_eye_coord_2 = single_face_landmarks.landmark[r_eye_idx[1]]

                # Left eye
                x1 = int(l_eye_coord_1.x * image_width)
                y1 = int(l_eye_coord_1.y * image_height)
                x2 = int(l_eye_coord_2.x * image_width)
                y2 = int(l_eye_coord_2.y * image_height)
                l_eye = img[y1:y2, x1:x2]
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # Right eye
                x1 = int(r_eye_coord_1.x * image_width)
                y1 = int(r_eye_coord_1.y * image_height)
                x2 = int(r_eye_coord_2.x * image_width)
                y2 = int(r_eye_coord_2.y * image_height)
                r_eye = img[y1:y2, x1:x2]
                #cv2.rectangle(img,(x1, y1), (x2, y2), (255, 255, 255), 1)
            return l_eye, r_eye  # 눈 크롭 이미지 반환
        else:
            return None



    def eye_location(self, img): #눈 앞머리 좌표
        x1, y1, x2, y2 = 0, 0, 0, 0
        image_width = self.img.shape[1]
        image_height = self.img.shape[0]
        l_eye_idx = 133
        r_eye_idx = 463
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            for single_face_landmarks in results.multi_face_landmarks:
                l_eye_coord= single_face_landmarks.landmark[l_eye_idx]
                r_eye_coord = single_face_landmarks.landmark[r_eye_idx]
                # Left eye
                x1 = int(l_eye_coord.x * image_width)
                y1 = int(l_eye_coord.y * image_height)

                # Right eye
                x2 = int(r_eye_coord.x * image_width)
                y2 = int(r_eye_coord.y * image_height)

        return (x1, y1), (x2, y2)

    def save_img(self, inputnum, index, faceNum, l_eye, r_eye):
        print('saved', faceNum)
        path = f'facial_image\\input_{inputnum}\\image_{index}\\{faceNum}'
        cv2.imwrite(path + '\\l_eye.jpg', l_eye)
        cv2.imwrite(path + '\\r_eye.jpg', r_eye)


