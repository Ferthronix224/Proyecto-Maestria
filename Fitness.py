# Tasa de repetibilidad
import cv2
import math
import numpy as np

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
def Flanned_Matcher(image, rotated_image, keypoints, rotated_keypoints, keypoints_number):
    if len(rotated_keypoints) > keypoints_number or len(keypoints) > keypoints_number or len(keypoints) == 0 or len(rotated_keypoints) == 0 or type(image) is int or type(rotated_image) is int:
        return 0, 0, 0
    original_keypoints_rotated = rotation(keypoints)
    distances = np.linalg.norm(original_keypoints_rotated[:, np.newaxis] - rotated_keypoints, axis=2)
    matches = distances <= 5
    good_matches = 0

    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rotated_image = np.uint8(rotated_image)
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)

    for i, match in enumerate(matches):
        if True in match:
            good_matches += 1
            x, y = keypoints[i].ravel().astype(int)
            image = cv2.circle(image, center=(y, x), radius=5, color=(0, 255, 0), thickness=-1)
        else:
            x, y = keypoints[i].ravel().astype(int)
            image = cv2.circle(image, center=(y, x), radius=5, color=(0, 0, 255), thickness=-1)

    for i, match in enumerate(matches.transpose()):
        if True in match:
            x, y = rotated_keypoints[i].ravel().astype(int)
            rotated_image = cv2.circle(rotated_image, center=(y, x), radius=5, color=(0, 255, 0), thickness=-1)
        else:
            x, y = rotated_keypoints[i].ravel().astype(int)
            rotated_image = cv2.circle(rotated_image, center=(y, x), radius=5, color=(0, 0, 255), thickness=-1)

    # Calcular tasa de repetibilidad
    if good_matches == 0:
        repeatability = 0
    else:
        repeatability = (good_matches / min(len(keypoints), len(rotated_keypoints))) * 100

    return repeatability, image, rotated_image

def rotation(image):
    image = image.astype(int)
    # Ángulo de rotación en grados
    angle_degrees = 5

    # Obtener las dimensiones de la imagen
    height, width = image.shape[:2]

    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Obtener la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

    image = np.float32(image)

    # Aplicar la rotación
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def distancia_euclidiana(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)