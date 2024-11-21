# Libraries
import cv2 
import time
import warnings
import numpy as np
# Scripts
from DE import Genotype
import Transformations as tr
import Process as pr

warnings.filterwarnings('ignore', category=RuntimeWarning)


# Main process for interest point detection
def main(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations, termination_criteria, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
    # Start timer
    start = time.time()
    # Condition in case the image was not found
    img_lists = [img1, img2]

    if any(img is None for img_list in img_lists for img in img_list):
        raise ValueError("Image not found")

    # Population creation through differential evolution
    genotype = Genotype(population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate)
    population = genotype.init_population()

    # A loop is performed over the established range of generations
    for generation in range(generations):
        repeatability_population, filter_M, images, transformed_images = pr.repeatability(population, img1.copy(), img2.copy(), umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        best_current_fitness = max(repeatability_population)
        best_current_genotype = filter_M[np.argmax(repeatability_population)]
        best_current_image = images[np.argmax(repeatability_population)]
        best_current_transformed_image = transformed_images[np.argmax(repeatability_population)]

        if generation == 0:
            best_fitness = best_current_fitness
            best_genotype = best_current_genotype
            best_image = best_current_image
            best_rotated_image = best_current_transformed_image
        else:
            if best_current_fitness > best_fitness:
                best_fitness = best_current_fitness
                best_genotype = best_current_genotype
                best_image = best_current_image
                best_rotated_image = best_current_transformed_image

        mutation = genotype.Mutation(population)
        crossover = genotype.Crossover(population, mutation)
        repeatability_crossover, _, _, _ = pr.repeatability(crossover, img1.copy(), img2.copy(), umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        genotype.Selection(population, crossover, repeatability_population, repeatability_crossover)

        # Stopping criterion when the best solution has been found
        if best_fitness >= termination_criteria:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')
            for i in range(len(images[1])):
                cv2.imwrite(f'img/results/original-{i+1}.jpg', best_image[i])
                cv2.imwrite(f'img/results/transformed-{i+1}.jpg', best_rotated_image[i])
            break
        # Console output when all generations have finished        
        elif generation == GENERATIONS - 1:
            print(f'Generation {GENERATIONS}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')
            for i in range(len(images[1])):
                cv2.imwrite(f'img/results/original-{i+1}.jpg', best_image[i])
                cv2.imwrite(f'img/results/transformed-{i+1}.jpg', best_rotated_image[i])

if __name__ == '__main__':
    # Parameters
    IMG1 = [cv2.imread(f'img/originals/{i}.jpg', cv2.IMREAD_GRAYSCALE)for i in range(1, 11)]
    IMG2 = [cv2.imread(f'img/rotation/{i}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(1, 11)]
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
    LOW_LIM_KN = 100  # KN -> Keypoints Number
    UP_LIM_KN = 2000
    TRANSFORMATION = tr.rotation
    TRANSFORMATION_VALUE = 30

    main(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM_GEN, UP_LIM_GEN, F, CROSSOVER_RATE, GENERATIONS, TERMINATION_CRITERIA, WR, LOW_LIM_KN, UP_LIM_KN, TRANSFORMATION, TRANSFORMATION_VALUE)