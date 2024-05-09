import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from flask import Flask, Response
from my_file_loader import load_shape_predictor_dat_file
from my_file_loader import load_example_video
import json

app = Flask(__name__)

host='localhost'
port=13967

@app.route('/video_feed')
def video_feed():
    return Response(capture())

mimetype = 'multipart/x-mixed-replace;boundary=frame'


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Инициализируем детектор лиц и предиктор ключевых точек
detector = dlib.get_frontal_face_detector()
# Вызовв функции загрузки файла с моделью для анализа лица
predictor = dlib.shape_predictor(
    load_shape_predictor_dat_file()
)  # Скачай файл по ссылке в официальной документации
def capture():
    # Загрузим изображение с камеры
    cap = cv2.VideoCapture(0) #load_example_video | 0
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break

        # Обнаружим лица на изображении
        faces = detector(gray)

        for face in faces:
            # Получим ключевые точки лица (включая рот)
            landmarks = predictor(gray, face)

            # Нарисуем выделение вокруг глаз
            # Извлечем координаты углов глаз
            left_eye_outer_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], np.int32)
            right_eye_outer_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], np.int32)

            for i in range(0, 67):
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=3, color=(0, 0, 255), thickness=-1)

            # Рассчитаем уровень закрытости глаз (1 - открыт, 0 - закрыт)
            left_eye_closure = eye_aspect_ratio(left_eye_outer_pts)
            right_eye_closure = eye_aspect_ratio(right_eye_outer_pts)

            # Отобразим уровень закрытости глаз на изображении
            cv2.putText(frame, f'Left Eye Closure: {left_eye_closure:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 0, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Right Eye Closure: {right_eye_closure:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 0, 210), 1, cv2.LINE_AA)

            # Рассчитаем вертикальные расстояния между верхней и нижней частью рта
            #mouth_height = np.linalg.norm(mouth_top_pts.mean(axis=0) - mouth_bottom_pts.mean(axis=0))
            mouth_height = np.linalg.norm(landmarks.part(51).y - landmarks.part(57).y)

            # Рассчитаем ширину рта (расстояние между уголками рта)
            mouth_width = np.linalg.norm(landmarks.part(48).x - landmarks.part(54).x)

            # Рассчитаем уровень открытости рта (1 - открыт, 0 - закрыт)
            mouth_openness = mouth_height / mouth_width

            # Отобразим уровень открытости рта на изображении
            cv2.putText(frame, f'Mouth Openness: {mouth_openness:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        (flag, encoded_image) = cv2.imencode('.jpeg', frame)
        yield (b'--frame\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

        # Выход из цикла при нажатии клавиши 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Освободим ресурсы
    cap.release()
    cv2.destroyAllWindows()

app.run(debug=True, threaded=True, host=host, port=port)