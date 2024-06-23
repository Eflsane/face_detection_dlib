import json
import socket
import cv2
import dlib
import imutils
import numpy as np
import torch
from imutils import face_utils
from torchsummary import summary

from face_calculations import get_head_rotations, get_eye_aspect_ratio, get_mouth_openness_value, get_smile_value
from face_detection_utils import XceptionNet, preprocess_image, FaceLandmarks, FaceLandmarksPart
from my_file_loader import load_shape_predictor_dat_file, load_model_file
from my_file_loader import load_example_video


def face_params():
    return get_face_params()


def connect_to_tcp_server(host, port):
    try:
        # connect to the port
        s.connect((host, port))
        print('connected to ' + host + ':' + str(port))
    except BaseException as ex:
        print(f'An exception has occurred. Details: \n {ex}')
        s.close()


def start_app(host='localhost', port=13967, frame_width=512, frame_height=512):

    try:
        connect_to_tcp_server(host, port)
        get_face_params(frame_width, frame_height)
    except BaseException as ex:
        print(f'An exception has occurred. Details: \n {ex}')
    finally:
        s.close()


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using the GPU")
else:
    device = torch.device("cpu")
    print("PyTorch is using the CPU")

predictor = XceptionNet()
predictor.load_state_dict(torch.load(load_model_file(), map_location=device))  # , map_location='cpu'

predictor.to(device)
predictor.eval()
print(summary(predictor, (1, 128, 128)))

detector = dlib.get_frontal_face_detector()

@torch.no_grad()
def get_face_params(frame_width=640, frame_height=480, camera=0):
    # Загрузим изображение с камеры
    cap = cv2.VideoCapture(camera)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    head_rot = np.array([0.0, 0.0, 0.0])
    left_eye_closure = 0.0
    right_eye_closure = 0.0
    mouth_openness = 0.0
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break

        # Обнаружим лица на изображении
        faces = detector(gray)

        for face in faces:
            landmarks = []
            # Нарисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Получим ключевые точки лица (включая рот)

            (x, y, w, h) = face_utils.rect_to_bb(face)

            crop_img = gray[y: y + h, x: x + w]
            preprocessed_image = preprocess_image(crop_img)
            landmarks_predictions = predictor(preprocessed_image)
            landmarks.append((landmarks_predictions, (x, y, h, w)))

            landmarks = landmarks_to_numerable_list(frame, landmarks)

            # landmarks = predictor(gray, face)

            if landmarks.total_parts() <= 0:
                break

            # Отобразим точки лица на изображении
            for i in range(0, 67):
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=3, color=(0, 0, 255), thickness=-1)

            # Извлечем координаты лица
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 67)],
                                       np.float32)

            # Расчет поворотов головы по трем осям
            head_rot = get_head_rotations(landmarks_array)
            # Отобразим уровень поворота головы по оси Х
            cv2.putText(frame, f'Head rot X: {head_rot[0]:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            # Отобразим уровень поворота головы по оси Y
            cv2.putText(frame, f'head rot Y: {head_rot[1]:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            # Отобразим уровень поворота головы по оси Z
            cv2.putText(frame, f'head rot Z: {head_rot[2]:.2f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Нарисуем выделение вокруг глаз
            # Извлечем координаты углов глаз
            left_eye_outer_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
                                          np.float32)
            right_eye_outer_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
                                           np.float32)
            # Рассчитаем уровень закрытости глаз (1 - открыт, 0 - закрыт)
            left_eye_closure = get_eye_aspect_ratio(left_eye_outer_pts)
            right_eye_closure = get_eye_aspect_ratio(right_eye_outer_pts)
            # Отобразим уровень закрытости глаз на изображении
            cv2.putText(frame, f'Left Eye Closure: {left_eye_closure:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 0, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Right Eye Closure: {right_eye_closure:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 0, 210), 1, cv2.LINE_AA)


            # Рассчитаем открытость рта
            # Извлечем координаты рта
            mouth_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 67)],
                                           np.float32)
            # Рассчитаем уровень открытости рта (1 - открыт, 0 - закрыт)
            mouth_openness = get_mouth_openness_value(landmarks_array)
            # Отобразим уровень открытости рта на изображении
            cv2.putText(frame, f'Mouth Openness: {mouth_openness:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Расчет силы улыбки
            # Рассчитаем силу улыбки, где 0 - нет улыбки, 1 - максимальная улыбка
            # smile_intensity = get_smile_value(landmarks_array)
            # Отобразим уровень улыбчивости
            # cv2.putText(frame, f'Im smiling: {smile_intensity:.2f}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 255), 1, cv2.LINE_AA)


        # Отобразим результат
        json_params = json.dumps({
            'headRotX': round(float(head_rot[0]), 2),
            'headRotY': round(float(head_rot[1]), 2),
            'headRotZ': round(float(head_rot[2]), 2),
            'leftEye': round(left_eye_closure, 2),
            'rightEye': round(right_eye_closure, 2),
            'mouthOpenness': round(mouth_openness, 2)
        })
        message = str(json_params) + '\n'
        s.send(message.encode())

        # Отобразим результат
        cv2.imshow('Face Detection (dlib)', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освободим ресурсы
    cap.release()
    cv2.destroyAllWindows()


def landmarks_to_numerable_list(image, faces_landmarks):
    image = image.copy()
    landmark_parts = FaceLandmarks()
    for landmarks, (left, top, height, width) in faces_landmarks:
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5)
        landmarks = landmarks.numpy()

        for i, (x, y) in enumerate(landmarks, 1):
            try:
                new_landmark_part = FaceLandmarksPart(int((x * width) + left), int((y * height) + top))
                landmark_parts.append_part(new_landmark_part.x, new_landmark_part.y)
                #cv2.circle(image, (new_landmark.x, new_landmark.y), 2, [40, 117, 255], -1)
                #
                # cv2.putText(image, str(i), (new_landmark.x, new_landmark.y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                #             (255, 255, 255), 1, cv2.LINE_AA)
            except:
                pass

    return landmark_parts


s = socket.socket()
start_app(frame_width=640, frame_height=480)
