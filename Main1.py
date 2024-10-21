# Libraries
import cv2 
import time
# Scripts
from DE import Genotype
import Transformations as tr
import Process as pr

# Main process for interest point detection
def main(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations, termination_criteria, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
    # Start timer
    start = time.time()
    # Condition in case the image was not found
    if img1 is None or img2 is None:
        raise ValueError("Image not found")

    # Dimension change
    # img1 = cv2.resize(img1, (500, 500))
    # img2 = cv2.resize(img2, (500, 500))

    # Grayscale conversion of the image
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Population creation through differential evolution
    genotype = Genotype(population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate)
    population = genotype.init_population()

    # A loop is performed over the established range of generations
    for generation in range(generations):        
        repeatability_population, filter_M, images, transformed_images = pr.repeatability(population, img1.copy(), img2.copy(), umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        best_current_fitness = max(repeatability_population)
        best_current_genotype = filter_M[repeatability_population.index(best_current_fitness)]
        best_current_image = images[repeatability_population.index(best_current_fitness)]
        best_current_transformed_image = transformed_images[repeatability_population.index(best_current_fitness)]

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

        # Console output with the best fitness every 10 generations
        if (generation + 1) % 10 == 0:
            print(f'Generation {generation + 1}: Best Fitness = {best_current_fitness}')
        # Stopping criterion when the best solution has been found
        if best_fitness >= termination_criteria:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness}')
            end = time.time()
            print(f'Time: {end - start}')
            cv2.imwrite(f'Original.jpg', best_image)
            cv2.imwrite(f'Transformed.jpg', best_rotated_image)
            break
        # Console output when all generations have finished        
        elif generation == GENERATIONS - 1:
            print(f'Generation {GENERATIONS}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness}')
            end = time.time()
            print(f'Time: {end - start}')
            cv2.imwrite(f'Original.jpg', best_image)
            cv2.imwrite(f'Transformed.jpg', best_rotated_image)

if __name__ == '__main__':
    # Parameters
    IMG1 = cv2.imread('img/img1.jpg')
    IMG2 = cv2.imread('img/Noisy1.jpg')
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
    UP_LIM_KN = 3000
    TRANSFORMATION = tr.translate
    TRANSFORMATION_VALUE = 1

    main(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM_GEN, UP_LIM_GEN, F, CROSSOVER_RATE, GENERATIONS, TERMINATION_CRITERIA, WR, LOW_LIM_KN, UP_LIM_KN, TRANSFORMATION, TRANSFORMATION_VALUE)