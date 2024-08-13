import cv2
import numpy as np
import MP
import DE
from Fitness import Flanned_Matcher
import Filters as ft

def detectar_puntos_de_interes(magnitud, umbral):
    # Crear una máscara booleana donde los elementos mayores que el umbral son True
    mask = magnitud > umbral

    # Encontrar los índices de los elementos que son True en la máscara
    indices = np.argwhere(mask)

    return indices

# Función para mostrar puntos de interés
def obtener_puntos_de_interes(imagen, puntos_de_interes, mostrar):
    for c in puntos_de_interes:
        x, y = c.ravel()
        imagen = cv2.circle(imagen, center=(y, x), radius=5, color=(0, 0, 255), thickness=-1)
    if mostrar:
        if imagen.dtype != np.uint8:
            print('int8')
            imagen = np.uint8(np.absolute(imagen))
        cv2.imshow('imagen', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return imagen

def normalizar(matriz):
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    matriz_normalizada = (matriz - min_val) / (max_val - min_val)
    return matriz_normalizada

def evaluation(img, filter):
    return eval(filter)

# Proceso principal de detección de puntos de interés
def deteccion_de_puntos_de_interes(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations):
    #  Condicion en caso de que la imagen no se halla encontrado
    if img1 is None or img2 is None:
        raise ValueError("Imagen no encontrada")

    # Cambio de dimensiones
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    # Conversion a escala de grises de la imagen
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Creacion del genotipo por medio de evolucion diferencial
    individual = DE.Individual(population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate)
    population = individual.init_population()

    for generation in range(generations):
        mutation = individual.Mutation(population)
        crossover = individual.Crossover(population, mutation)

        # Proceso de mapeo
        filter_MP = MP.generate(population)
        print(filter_MP)

        # Evaluacion de los filtros
        # Cambiar el argumento de filter_MP si se quiere un filtro en especifico
        filtro1 = evaluation(img1.copy(), filter_MP)
        filtro2 = evaluation(img2.copy(), filter_MP)

        # Normalización de los datos
        filtro1_normalizada = normalizar(filtro1)
        filtro2_normalizada = normalizar(filtro2)

        # Detectar puntos de interés basados en la magnitud
        puntos_de_interes_1 = detectar_puntos_de_interes(filtro1_normalizada, umbral_deteccion)
        puntos_de_interes_2 = detectar_puntos_de_interes(filtro2_normalizada, umbral_deteccion)

        # Mostrar los puntos de interés detectados
        imagen1 = obtener_puntos_de_interes(img1, puntos_de_interes_1, False)
        imagen2 = obtener_puntos_de_interes(img2, puntos_de_interes_2, False)

        output, repeatability = Flanned_Matcher(imagen1, imagen2)
        individual.Selection(population, crossover)

        # Imprime la tasa de repetibilidad
        print(f'Repeatability rate: {repeatability:.2f}%')

        # Muestra la imagen
        cv2.imshow('Match.jpg', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parámetros
    IMG1 = cv2.imread('img/Cuadrado 3.JPG')
    IMG2 = cv2.imread('img/Formas.png')
    UMBRAL = 0.95
    POPULATION_SIZE = 50
    GENOTYPE_LENGTH = 50
    LOW_LIM = 1
    UP_LIM = 255
    MUTATION_RATE = 0.5
    CROSSOVER_RATE = 0.7
    GENERATIONS = 100

    deteccion_de_puntos_de_interes(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM, UP_LIM, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS)
'''
for i in range(generations):
    fitness_value = [fit(x) for x in population]
    fitness_value.sort()
    best_individual = fitness_value[0]
    print(i, best_individual)
    if best_individual <= termination_criteria:
        break
    mutation = Mutation(F, population)
    crossover = Crossover(CR, population, mutation)
    Selection(population, crossover, fit)
'''