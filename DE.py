# Algoritmo de evolución diferencial
import numpy as np

class Individual:
    def __init__(self, dimension, low_lim, up_lim):
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.dimension = dimension
        self.genotype = self.process()

    def process(self):
        return np.random.uniform(low=self.low_lim, high=self.up_lim, size=self.dimension)

    def return_genotype(self):
        return self.genotype

def Mutation(F, population):
    mutated_population = []
    for i in range(len(population)):
        indexes = np.arange(0, len(population))
        indexes = np.delete(indexes, i)
        new = np.random.choice(indexes, 3, replace=False)
        mutated_population.append(population[new[2]] + (F * (population[new[1]] - population[new[0]])))
    return mutated_population

def Crossover(CR, population, mutated_population):
    cross_population = []
    for i in range(len(mutated_population)):
        cross_population.append([])
        for j in range(len(mutated_population[0])):
            if np.random.uniform(0, 1) <= CR:
                cross_population[i].append(mutated_population[i][j])
            else:
                cross_population[i].append(population[i][j])
    return np.array(cross_population)

def Selection(population, cross_population, fit, x=None, y=None):
    if y is not None and x is not None:
        population_fitness = [fit(x, y) for _ in population]
        cross_population_fitness = [fit(x, y) for _ in cross_population]
    else:
        population_fitness = [fit(x) for x in population]
        cross_population_fitness = [fit(x) for x in cross_population]

    for i in range(len(population)):
        if cross_population_fitness[i] <= population_fitness[i]:
            population[i] = cross_population[i]