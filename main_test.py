import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from my_file_loader import load_shape_predictor_dat_file

# Инициализируем детектор лиц и предиктор ключевых точек
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    load_shape_predictor_dat_file()
)  # Скачать файл по ссылке в официальной документации


def head_rotations(face):
    # Расчет поворотов головы по трем осям
    # Расчет поворота головы по оси Х
    # Рассчитаем расстояние от носа до правой части лица
    # face_right_pt = np.array([(landmarks.part(16).x, landmarks.part(16).y)],
    #                         np.float32)
    # nose_center_pt = np.array([(landmarks.part(30).x, landmarks.part(30).y)],
    #                         np.float32)
    # nose_to_right = dist.euclidean(face_right_pt[0], nose_center_pt[0])
    nose_to_right = np.abs(face[16][0] - face[30][0])

    # Рассчитаем расстояние от носа до левой части лица
    # face_left_pt = np.array([(landmarks.part(0).x, landmarks.part(0).y)],
    #                         np.float32)
    # nose_to_left = dist.euclidean(face_left_pt[0], nose_center_pt[0])
    nose_to_left = np.abs(face[0][0] - face[30][0])

    # Рассчитаем расстояние от правой до левой части лица
    # right_to_left = dist.euclidean(face_right_pt[0], face_left_pt[0])
    right_to_left = np.abs(face[16][0] - face[0][0])

    # Рассчитаем уровень поворота головы по оси Х (1 - направо, -1 - налево)
    head_rot_X = (nose_to_left - nose_to_right) / right_to_left

    # Расчет поворота головы по оси У
    # Рассчитаем расстояние от носа до нижней до верхней части носа
    nose_to_top = np.abs(face[33][1] - face[27][1])

    # Рассчитаем расстояние от подбородка до нижней части носа
    nose_to_bottom = np.abs(face[8][1] - face[33][1])

    # Рассчитаем расстояние от подбородка до верхней части носа
    top_to_bottom = np.abs(face[8][1] - face[27][1])

    # Рассчитаем уровень поворота головы по оси У (1 - вверх, -1 - вниз)
    head_rot_Y = (nose_to_bottom - nose_to_top) / top_to_bottom

    # Расчет поворота головы по оси Z
    # Для рассчетов беруться крайние точки глаз
    left_eye_y = face[39][1]
    right_eye_y = face[42][1]

    # Рассчитаем уровень поворота головы по оси Z (1 - враво, -1 - влево)
    head_rot_Z = (-1 * (right_eye_y - left_eye_y)
                     / (face[31][0] - face[35][1]))
    return np.array([head_rot_X, head_rot_Y, head_rot_Z])


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


def mouth_openness_value(mouth):
    # Рассчитаем открытость рта
    # Рассчитаем вертикальные расстояния между верхней и нижней частью рта
    mouth_top_pt = mouth[62]
    mouth_bottom_pt = mouth[66]
    mouth_height = dist.euclidean(mouth_top_pt, mouth_bottom_pt)

    # Рассчитаем ширину рта (расстояние между уголками рта)
    mouth_left_pt = mouth[60]
    mouth_right_pt = mouth[64]
    mouth_width = dist.euclidean(mouth_left_pt, mouth_right_pt)

    # Рассчитаем уровень открытости рта (1 - открыт, 0 - закрыт)
    return mouth_height / mouth_width


def smile_value(mouth):
    # Расчет силы улыбки
    # Рассчитаем ширину рта (расстояние между уголками рта)
    mouth_left_outer_pt = mouth[48]
    mouth_right_outer_pt = mouth[54]
    mouth_outer_width = dist.euclidean(mouth_left_outer_pt, mouth_right_outer_pt)

    # рассчитаем расстояние между верхней губой и линией соединяющей углы рта
    mouth_top_left_pt = mouth[50]
    mouth_top_right_pt = mouth[52]
    mouth_bottom_left_pt = mouth[58]
    top_lip_distance = dist.cdist([mouth_top_right_pt],
                                  [mouth_top_left_pt, mouth_bottom_left_pt],
                                  'euclidean')[0, 1]

    # Рассчитаем силу улыбки, где 0 - нет улыбки, 1 - максимальная улыбка
    return min(1.0, top_lip_distance / mouth_outer_width)


def brow_raising(brow):
    print()


frame_width = 1366
frame_height = 768

def capture():
    # Загрузим изображение с камеры
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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

            # Нарисуем прямоугольник вокруг лица
            # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Отобразим точки лица на изображении
            for i in range(0, 67):
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), radius=3, color=(0, 0, 255), thickness=-1)

            # Извлечем координаты лица
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 67)],
                                       np.float32)

            # Расчет поворотов головы по трем осям
            head_rot = head_rotations(landmarks_array)
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
            left_eye_closure = eye_aspect_ratio(left_eye_outer_pts)
            right_eye_closure = eye_aspect_ratio(right_eye_outer_pts)
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
            mouth_openness = mouth_openness_value(landmarks_array)
            # Отобразим уровень открытости рта на изображении
            cv2.putText(frame, f'Mouth Openness: {mouth_openness:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Расчет силы улыбки
            # Рассчитаем силу улыбки, где 0 - нет улыбки, 1 - максимальная улыбка
            smile_intensity = smile_value(landmarks_array)
            # Отобразим уровень улыбчивости
            cv2.putText(frame, f'Im smiling: {smile_intensity:.2f}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)




        # Отобразим результат
        cv2.imshow('Face Detection (dlib)', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освободим ресурсы
    cap.release()
    cv2.destroyAllWindows()

capture()