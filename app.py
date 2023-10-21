from flask import Flask, request, send_file
import cv2


app = Flask(__name__)


@app.route('/')
def nothing():
    return 'api_test 폴더에 api 사용 예시 파이썬 코드가 나와있습니다'


@app.route('/overlay', methods=['POST', 'GET'])
def overlay():
    print(request.method)
    if request.method=='POST':
        f = request.files['main']
        filepath_main = './save_image/overlay_background.png'
        f.save(filepath_main)
        f = request.files['overlay']
        filepath_overlay = './save_image/overlay.png'
        f.save(filepath_overlay)

        image_main = cv2.imread(filepath_main)
        image_overlay = cv2.imread(filepath_overlay)

        pos_topleft, pos_bottomright = request.form['topleft'], request.form['bottomright']
        pos_topleft=tuple(map(int, pos_topleft.split()))
        pos_bottomright=tuple(map(int, pos_bottomright.split()))
        size = (pos_bottomright[0] - pos_topleft[0], pos_bottomright[1] - pos_topleft[1])

        image_overlay = cv2.resize(image_overlay, size)
        for i in range(size[0]):
            for j in range(size[1]):
                image_main[pos_topleft[1] + j][pos_topleft[0] + i] = image_overlay[j][i]
        cv2.imwrite('./save_image/result.png', image_main)
        return send_file('./save_image/result.png', mimetype='image/png')
    elif request.method=='GET':
        return send_file('./save_image/result.png', mimetype='image/png')





if __name__ == '__main__':
    app.run()
