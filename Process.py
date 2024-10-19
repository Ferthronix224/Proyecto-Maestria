import numpy as np
from MP import MP
from Fitness import Fitness
import Filters as ft

def detectar_puntos_de_interes(magnitud, umbral):
    # Crear una máscara booleana donde los elementos mayores que el umbral son True
    mask = magnitud > umbral

    # Encontrar los índices de los elementos que son True en la máscara
    indices = np.argwhere(mask)

    return indices

def normalizar(matriz):
    if type(matriz) == int:
        return 0
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    if min_val == np.nan or max_val == np.nan or max_val == 0.0 or min_val == -np.inf or max_val == np.inf:
        return 0
    matriz_normalizada = (matriz - min_val) / (max_val - min_val)
    return matriz_normalizada

def evaluation(img, filter):
    if filter == 'Worst':
        return 0
    return eval(filter)

def repeatability(population, img1, img2, umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
    # Proceso de mapeo
    filter_MP = [MP().generate(population[i], wr) for i in range(len(population))]
    # filter_MP = ['ft.Gau1(img-ft.Log(ft.Gau1(ft.Lap(ft.Gau2(ft.Gau1(ft.Gau2(ft.Gau1(img))+ft.Gau2(ft.Gau2(img)))/ft.LapG1(img))))))']
    
    img1 = np.where(img1 == 0, 1e-4, img1)
    img2 = np.where(img2 == 0, 1e-4, img2)
    # Evaluacion de los filtros
    # Cambiar el argumento de filter_MP si se quiere un filtro en especifico
    filtro1 = [evaluation(img1.copy(), filter_MP[i]) for i in range(len(filter_MP))]
    filtro2 = [evaluation(img2.copy(), filter_MP[i]) for i in range(len(filter_MP))]

    # Normalización de los datos
    filtro1_normalizada = [normalizar(filtro1[i]) for i in range(len(filtro1))]
    filtro2_normalizada = [normalizar(filtro2[i]) for i in range(len(filtro2))]

    # Detectar puntos de interés basados en la magnitud
    puntos_de_interes_1 = [detectar_puntos_de_interes(filtro1_normalizada[i], umbral_deteccion) for i in
                           range(len(filtro1_normalizada))]
    puntos_de_interes_2 = [detectar_puntos_de_interes(filtro2_normalizada[i], umbral_deteccion) for i in
                           range(len(filtro2_normalizada))]

    repeatability, images, rotated_images = zip(*[Fitness(img1.copy(), img2.copy(), puntos_de_interes_1[i], puntos_de_interes_2[i], low_keypoints_number, up_keypoints_number, transformation, transformation_value).process() for i in range(len(filter_MP))])
    
    return repeatability, filter_MP, images, rotated_images