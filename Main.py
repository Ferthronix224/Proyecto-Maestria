import cv2
import numpy as np
import MP
import DE
from Fitness import Flanned_Matcher
import Filters as ft
import time
import Transformations as tr

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
    matriz_normalizada = (matriz - min_val) / (max_val - min_val)
    return matriz_normalizada

def evaluation(img, filter):
    if filter == 'Worst':
        return 0
    return eval(filter)

def repetibilidad(population, img1, img2, umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
    # Proceso de mapeo
    filter_MP = [MP.generate(population[i], wr) for i in range(len(population))]

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

    repeatability, images, rotated_images = zip(*[Flanned_Matcher(img1.copy(), img2.copy(), puntos_de_interes_1[i], puntos_de_interes_2[i], low_keypoints_number, up_keypoints_number, transformation, transformation_value) for i in range(len(filter_MP))])

    return repeatability, filter_MP, images, rotated_images

# Proceso principal de detección de puntos de interés
def deteccion_de_puntos_de_interes(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations, termination_criteria, wr, low_keypoints_number, up_keypoints_number, no, transformation, transformation_value):
    inicio = time.time()
    #  Condicion en caso de que la imagen no se halla encontrado
    if img1 is None or img2 is None:
        raise ValueError("Imagen no encontrada")

    # Cambio de dimensiones
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    # Conversion a escala de grises de la imagen
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Creacion de la población por medio de evolucion diferencial
    individual = DE.Individual(population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate)
    population = individual.init_population()

    # Se hace un ciclo con el rango de las generaciones establecidas
    for generation in range(generations):
        repeatability_population, filter_M, images, rotated_images = repetibilidad(population, img1.copy(), img2.copy(), umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        best_current_fitness = max(repeatability_population)
        best_current_genotype = filter_M[repeatability_population.index(best_current_fitness)]
        best_current_image = images[repeatability_population.index(best_current_fitness)]
        best_current_rotated_image = rotated_images[repeatability_population.index(best_current_fitness)]

        if generation == 0:
            best_fitness = best_current_fitness
            best_genotype = best_current_genotype
            best_image = best_current_image
            best_rotated_image = best_current_rotated_image
        else:
            if best_current_fitness > best_fitness:
                best_fitness = best_current_fitness
                best_genotype = best_current_genotype
                best_image = best_current_image
                best_rotated_image = best_current_rotated_image

        mutation = individual.Mutation(population)
        crossover = individual.Crossover(population, mutation)
        repeatability_crossover, _, _, _ = repetibilidad(crossover, img1.copy(), img2.copy(), umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        individual.Selection(population, crossover, repeatability_population, repeatability_crossover)

        # Impresión de pantalla con el mejor fitness cada 100 generaciones
        if (generation + 1) % 10 == 0:
            print(f'Generation {generation + 1}: Best Fitness = {best_current_fitness}')
        # Criterio de paro cuando ya se encontró la mejor solución
        if best_fitness >= termination_criteria:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness}')
            fin = time.time()
            print(f'Tiempo: {fin - inicio}')
            cv2.imwrite(f'Original{no}.jpg', best_image)
            cv2.imwrite(f'Transformed{no}.jpg', best_rotated_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            break
        # Impresión de pantalla cuando ya se terminaron las generaciones
        elif generation == GENERATIONS - 1:
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness}')
            fin = time.time()
            print(f'Tiempo: {fin - inicio}')
            cv2.imwrite(f'Original{no}.jpg', best_image)
            cv2.imwrite(f'Transformed{no}.jpg', best_rotated_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

for i in range(1, 6):
    if __name__ == '__main__':
        # Parámetros
        IMG1 = cv2.imread(f'img/Imagen{i}.jpg')
        IMG2 = cv2.imread(f'img/Rotation{i}.JPG')
        UMBRAL = 0.95
        POPULATION_SIZE = 20
        GENOTYPE_LENGTH = 50
        LOW_LIM_GEN = 1
        UP_LIM_GEN = 255
        F = 0.5  # Xm = Xi + f (x2 - x3)
        CROSSOVER_RATE = 0.7
        GENERATIONS = 100
        TERMINATION_CRITERIA = 95.0
        WR = 3
        LOW_LIM_KN = 10  # KN -> Keypoints Number
        UP_LIM_KN = 3000
        no = i
        TRANSFORMATION = tr.rotation
        TRANSFORMATION_VALUE = 45

        deteccion_de_puntos_de_interes(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM_GEN, UP_LIM_GEN, F, CROSSOVER_RATE, GENERATIONS, TERMINATION_CRITERIA, WR, LOW_LIM_KN, UP_LIM_KN, no, TRANSFORMATION, TRANSFORMATION_VALUE)