# Libraries
from skimage import io
import time
import warnings
import cupy as cp
# Scripts
from DE import Genotype
from Filters import Filters

from Transformations import Transformations
from Process import Process

warnings.filterwarnings('ignore', category=RuntimeWarning)
tr = Transformations()
ft = Filters()
pr = Process()

# Main process for interest point detection
def main(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
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
        repeatability_population, filter_M = pr.repeatability(population, img1, img2, umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        best_current_fitness = max(repeatability_population)
        best_current_genotype = filter_M[int(repeatability_population.index(best_current_fitness))]

        if generation == 0:
            best_fitness = best_current_fitness
            best_genotype = best_current_genotype
        else:
            if best_current_fitness > best_fitness:
                best_fitness = best_current_fitness
                best_genotype = best_current_genotype

        mutation = genotype.Mutation(population)
        crossover = genotype.Crossover(population, mutation)
        repeatability_crossover, _ = pr.repeatability(crossover, img1, img2, umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value)
        genotype.Selection(population, crossover, repeatability_population, repeatability_crossover)

        # Stopping criterion when the best solution has been found
        if generation == 0:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')

        if generation % 10 == 0:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')

        if best_fitness >= umbral_deteccion * 100:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')
            break
        # Console output when all generations have finished
        elif generation == GENERATIONS - 1:
            print(f'Generation {GENERATIONS}')
            print(f'Best Solution: {best_genotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')

rutes = ['rotated/30', 'rotated/60', 'rotated/90', 'scale/50', 'scale/70', 'scale/90', 'translated/10', 'translated/20', 'translated/30']
transformations = [tr.rotation, tr.rotation, tr.rotation, tr.scale, tr.scale, tr.scale, tr.translate, tr.translate, tr.translate]
transformations_values = [30, 60, 90, 50, 70, 90, 10, 20, 30]

for index in range(9):
    # Parameters
    IMG1 = [cp.asarray(io.imread(f'drive/MyDrive/img/originals/{i}.jpg', True)) for i in range(1, 293)]
    IMG2 = [cp.asarray(io.imread(f'drive/MyDrive/img/{rutes[index]}/{i}.jpg', True)) for i in range(1, 293)]

    UMBRAL = 0.90
    POPULATION_SIZE = 30
    GENOTYPE_LENGTH = 50
    LOW_LIM_GEN = 1
    UP_LIM_GEN = 255
    F = 0.5  # Xm = Xi + f (x2 - x3)
    CROSSOVER_RATE = 0.7
    GENERATIONS = 100
    WR = 3
    LOW_LIM_KN = 10  # KN -> Keypoints Number
    UP_LIM_KN = 5000
    TRANSFORMATION = transformations[index]
    TRANSFORMATION_VALUE = transformations_values[index]

    main(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM_GEN, UP_LIM_GEN, F, CROSSOVER_RATE, GENERATIONS, WR, LOW_LIM_KN, UP_LIM_KN, TRANSFORMATION, TRANSFORMATION_VALUE)