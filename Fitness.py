# Tasa de repetibilidad
import cv2
import math
import numpy as np

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
def Flanned_Matcher(image1, rotated_keypoints):
    mask = image1 > 0.95

    # Encontrar los índices de los elementos que son True en la máscara

    image_rotation = rotation(mask)

    indices = np.argwhere(mask)

    good_matches = 0

    for indices in image_rotation:
        for coord2 in rotated_keypoints:
            if distancia_euclidiana(indices, coord2) < 5:
                good_matches += 1

    # Calcular tasa de repetibilidad
    if min(len(rotated_keypoints), len(rotated_keypoints)) == 0:
        repeatability = 0
    else:
        repeatability = good_matches / min(len(rotated_keypoints), len(rotated_keypoints)) * 100

    return repeatability

def rotation(matrix):
    # Ángulo de rotación en grados
    angle_degrees = 15

    # Obtener las dimensiones de la imagen
    height, width = matrix.shape[:2]

    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Obtener la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

    # Aplicar la rotación
    return cv2.warpAffine(matrix, rotation_matrix, (width, height))

def distancia_euclidiana(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)