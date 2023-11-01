import os
from cropFace import Face
from cropEye import Eye
from classify import classify_img

# 얼굴 크롭 후 저장, 사람마다 눈 크롭
main_folder = '..\input'

def readfile():
    closed_people = [] #감은 사람 id
    eye_points = [] # 감은 사람 눈 왼,오 좌표 ((x1, y1), (x2, y2))

    for inputnum, item in enumerate(os.listdir(main_folder)):
        sub_folder = os.path.join(main_folder, item) #E:\2023\2023_1_1\comp\input\1
        if os.path.isdir(sub_folder):
            for index, image_name in enumerate(os.listdir(sub_folder)): #index는 input\1 내 파일 번호
                image_path = sub_folder + f'\{image_name}'

                face = Face(image_path)
                eye = Eye(image_path)

                sorted_people = face.detect(show_results=False)
                if sorted_people == None:
                    print('No person detected')
                else:
                    for person in sorted_people:
                        faceNum, facial_img = person[0], person[1]
                        if eye.cropEye(facial_img) == None:
                            print('No Eye detected')
                            eye_points.append(None)
                        else:
                            l_eye, r_eye = eye.cropEye(facial_img)

                            face.save_img(inputnum, index, faceNum, facial_img)
                            eye.save_img(inputnum, index, faceNum, l_eye, r_eye)

                            if classify_img(facial_img) == 'Opened':
                                continue
                            else:
                                closed_people.append(faceNum)
                                eye_points.append((eye.eye_location(facial_img)))

    return closed_people, eye_points




