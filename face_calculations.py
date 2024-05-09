import numpy as np
from scipy.spatial import distance as dist


def get_head_rotations(face):
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
    eye_dist_y = right_eye_y - left_eye_y
    eye_dist_xy = dist.euclidean(face[42], face[39])

    # Рассчитаем уровень поворота головы по оси Z (1 - враво, -1 - влево)
    # head_rot_Z = (-1 * (right_eye_y - left_eye_y)
    #                  / (face[31][0] - face[35][1]))
    head_rot_Z = eye_dist_y / eye_dist_xy
    return np.array([head_rot_X, head_rot_Y, head_rot_Z])


def get_eye_aspect_ratio(eye) -> float:
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


def get_mouth_openness_value(face) -> float:
    # Рассчитаем открытость рта
    # Рассчитаем вертикальные расстояния между верхней и нижней частью рта
    mouth_top_pt = face[62]
    mouth_bottom_pt = face[66]
    mouth_height = dist.euclidean(mouth_top_pt, mouth_bottom_pt)

    # Рассчитаем ширину рта (расстояние между уголками рта)
    mouth_left_pt = face[60]
    mouth_right_pt = face[64]
    mouth_width = dist.euclidean(mouth_left_pt, mouth_right_pt)

    # Рассчитаем уровень открытости рта (1 - открыт, 0 - закрыт)
    return mouth_height / mouth_width


def get_smile_value(face) -> float:
    # Расчет силы улыбки
    # Рассчитаем ширину рта (расстояние между уголками рта)
    mouth_left_outer_pt = face[48]
    mouth_right_outer_pt = face[54]
    mouth_outer_width = dist.euclidean(mouth_left_outer_pt, mouth_right_outer_pt)

    # рассчитаем расстояние между верхней губой и линией соединяющей углы рта
    mouth_top_left_pt = face[50]
    mouth_top_right_pt = face[52]
    mouth_bottom_left_pt = face[58]
    top_lip_distance = dist.cdist([mouth_top_right_pt],
                                  [mouth_top_left_pt, mouth_bottom_left_pt],
                                  'euclidean')[0, 1]

    # Рассчитаем силу улыбки, где 0 - нет улыбки, 1 - максимальная улыбка
    return min(1.0, top_lip_distance / mouth_outer_width)


def get_brows_values(face):
    print("NOT IMPLEMENTED YET")
