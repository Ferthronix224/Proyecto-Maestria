# Libraries.
import cupy as cp
from skimage import io
import time
import warnings
# Scripts.
from Transformations import Transformations
from Filters import Filters
from Process import Process
from Genotype import Genotype

# Configuration to ignore irrelevant information in console.
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Instances.
tr = Transformations()
ft = Filters()
pr = Process()

# Main process for interest point detection.
def main(img1, img2, umbral_deteccion, population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate, generations, wr, low_interest_points_number, up_interest_points_number, transformation, transformation_value):
    # Start timer.
    start = time.time()

    # Condition in case some image was not found.
    img_lists = [img1, img2]    
    if any(img is None for img_list in img_lists for img in img_list):
        raise ValueError("Image not found")

    # Genotype object.
    genotype = Genotype(population_size, genotype_length, low_lim, up_lim, mutation_rate, crossover_rate)
    
    # Population creation through differential evolution.
    population = genotype.init_population()

    # A loop is performed over the established range of generations.
    for generation in range(generations):
        # Repeatability population and its corresponding filters.
        repeatability_population, filter_M = pr.repeatability(population, img1, img2, umbral_deteccion, wr, low_interest_points_number, up_interest_points_number, transformation, transformation_value)
        best_current_fitness = max(repeatability_population)
        best_current_phenotype = filter_M[int(repeatability_population.index(best_current_fitness))]
        changes = False

        # Condition to save the best fitness and phenotype.
        if generation == 0:
            best_fitness = best_current_fitness
            best_phenotype = best_current_phenotype
        else:
            if best_current_fitness > best_fitness:
                best_fitness = best_current_fitness
                best_phenotype = best_current_phenotype
                changes = True

        # Evolutionary Process.
        mutation = genotype.Mutation(population)
        crossover = genotype.Crossover(population, mutation)
        repeatability_crossover, _ = pr.repeatability(crossover, img1, img2, umbral_deteccion, wr, low_interest_points_number, up_interest_points_number, transformation, transformation_value)
        genotype.Selection(population, crossover, repeatability_population, repeatability_crossover)

        # Condition to display information in first generation.
        if generation == 0:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_phenotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')

        # Condition to display information if there is any change.
        if changes:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_phenotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')
            
        # Condition to display information if there is any change.
        if best_fitness >= umbral_deteccion * 100:
            print(f'Generation {generation + 1}')
            print(f'Best Solution: {best_phenotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')
            break
        # Console output when all generations have finished.
        elif generation == GENERATIONS - 1:
            print(f'Generation {GENERATIONS}')
            print(f'Best Solution: {best_phenotype}')
            print(f'Best Fitness: {best_fitness:.2f}%')
            end = time.time()
            print(f'Time: {end - start:.2f} segundos')

# Available configurations.
rutes = ['rotated/90', 'rotated/180', 'rotated/270', 'translated/10', 'translated/20', 'translated/30', 'scale/50', 'scale/70', 'scale/90']
transformations = [tr.rotation, tr.rotation, tr.rotation, tr.translate, tr.translate, tr.translate, tr.scale, tr.scale, tr.scale]
transformations_values = [90, 180, 270, 10, 20, 30, 50, 70, 90]


mrange = range(1, 293)
# index = -1
# for _ in range(1):
for index in range(len(rutes)):
    print(rutes[index])
    IMG1 = [cp.asarray(io.imread(f'img/originals/{i}.jpg', True)) for i in mrange]
    IMG2 = [cp.asarray(io.imread(f'img/{rutes[index]}/{i}.jpg', True)) for i in mrange]

    UMBRAL = 0.95
    POPULATION_SIZE = 20
    GENOTYPE_LENGTH = 50
    LOW_LIM_GEN = 1
    UP_LIM_GEN = 255
    F = 0.5  # Xm = Xi + f (x2 - x3).
    CROSSOVER_RATE = 0.7
    GENERATIONS = 100
    WR = 3
    LOW_LIM_IPN = 10  # IPN -> Interest Points Number.
    UP_LIM_IPN = 7000
    TRANSFORMATION = transformations[index]
    TRANSFORMATION_VALUE = transformations_values[index]

    main(IMG1, IMG2, UMBRAL, POPULATION_SIZE, GENOTYPE_LENGTH, LOW_LIM_GEN, UP_LIM_GEN, F, CROSSOVER_RATE, GENERATIONS, WR, LOW_LIM_IPN, UP_LIM_IPN, TRANSFORMATION, TRANSFORMATION_VALUE)