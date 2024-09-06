# Tasa de repetibilidad
import cv2
import math
import numpy as np

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
def Flanned_Matcher(keypoints, rotated_keypoints, keypoints_number):
    if len(rotated_keypoints) > keypoints_number or len(keypoints) > keypoints_number or len(keypoints) == 0 or len(rotated_keypoints) == 0:
        return 0

    keypoints = rotation(keypoints)

    distances = np.linalg.norm(keypoints[:, np.newaxis] - rotated_keypoints, axis=2)

    matches = distances <= 5

    good_matches = 0

    for i in matches:
        if True in i:
            good_matches += 1

    # Calcular tasa de repetibilidad
    if good_matches == 0:
        repeatability = 0
    else:
        repeatability = (good_matches / min(len(keypoints), len(rotated_keypoints))) * 100
        print(repeatability)

    return repeatability

def rotation(image):
    image = image.astype(int)
    # Ángulo de rotación en grados
    angle_degrees = 15

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