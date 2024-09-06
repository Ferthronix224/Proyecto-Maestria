# Tasa de repetibilidad
import cv2
import math
import numpy as np

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
def Flanned_Matcher(image1, rotated_keypoints):
    if type(image1) == int:
        return 0
    if len(rotated_keypoints) > 10_000 or len(image1) > 10_000 or len(image1) == 0 or len(rotated_keypoints) == 0:
        return 0

    mask = image1 > 0.95

    # Encontrar los índices de los elementos que son True en la máscara

    image_rotation = rotation(mask)

    indices = np.argwhere(image_rotation)

    distances = np.linalg.norm(indices[:, np.newaxis] - rotated_keypoints, axis=2)
    good_matches = np.sum(distances < 5)

    print('exitos de ' + str(good_matches) + ' minimo de ' + str(min(len(indices), len(rotated_keypoints))))
    # Calcular tasa de repetibilidad
    if good_matches == 0:
        repeatability = 0
    else:
        repeatability = (good_matches / min(indices.size, rotated_keypoints.size)) * 100
        print(repeatability)

    return repeatability

def rotation(matrix):
    matrix = matrix.astype(int)
    # Ángulo de rotación en grados
    angle_degrees = 15

    # Obtener las dimensiones de la imagen
    height, width = matrix.shape[:2]

    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Obtener la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

    matrix = np.float32(matrix)

    # Aplicar la rotación
    return cv2.warpAffine(matrix, rotation_matrix, (width, height))

def distancia_euclidiana(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)