# Algoritmo de evolución diferencial
import numpy as np


class Individual:
    def __init__(self, population_size, dimension, low_lim, up_lim, MR, CR):
        self.population_size = population_size
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.dimension = dimension
        self.CR = CR
        self.MR = MR

    def random_genotype(self, low_lim, up_lim, dimension):
        return np.random.uniform(low=low_lim, high=up_lim, size=dimension)

    def init_population(self):
        return [self.random_genotype(self.low_lim, self.up_lim, self.dimension) for _ in range(self.population_size)]

    def Mutation(self, population):
        mutated_population = []
        for i in range(len(population)):
            indexes = np.arange(0, len(population))
            indexes = np.delete(indexes, i)
            new = np.random.choice(indexes, 3, replace=False)
            mutated_population.append(population[new[2]] + (self.MR * (population[new[1]] - population[new[0]])))
        return mutated_population

    def Crossover(self, population, mutated_population):
        cross_population = []
        for i in range(len(mutated_population)):
            cross_population.append([])
            for j in range(len(mutated_population[0])):
                if np.random.uniform(0, 1) <= self.CR:
                    cross_population[i].append(mutated_population[i][j])
                else:
                    cross_population[i].append(population[i][j])
        return np.array(cross_population)

    def Selection(self, population, cross_population, population_fitness, crossover_fitness):
        for i in range(len(population)):
            if crossover_fitness[i] >= population_fitness[i]:
                population[i] = cross_population[i]
