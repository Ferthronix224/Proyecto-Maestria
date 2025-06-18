import numpy as np
import re

class Genotype:
    """
    Class Genotype to aplicate Differential Evolution.

    Parameters
    ---------
    population_size (int): Number of individuals.
    dimension (int): Number of codons in a individual.
    low_lim (int): Lower value in a codon.
    up_lim (int): Upper value in a codon.
    F (float): Constant requiere in Mutation fuction.
    CR (float): Crossover rate.
    """

    # Constructor.
    def __init__(self, population_size, dimension, low_lim, up_lim, F, CR):
        self.population_size = population_size
        self.dimension = dimension
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.F = F
        self.CR = CR

    # Function to make an individual.
    def random_genotype(self, low_lim, up_lim, dimension):
        return np.random.randint(low=low_lim, high=up_lim, size=dimension)

    # Function to make a population.
    def init_population(self):
        return [self.random_genotype(self.low_lim, self.up_lim, self.dimension) for _ in range(self.population_size)]

    # Function to mutate the population.
    def Mutation(self, population):
        mutated_population = []
        for i in range(len(population)):
            indexes = np.arange(0, len(population))
            indexes = indexes[indexes != i]
            new = np.random.choice(indexes, 3, replace=False)
            mutated_population.append(population[2] + (self.F * (population[int(new[1])] - population[int(new[0])])))
        return mutated_population
    
    # Function to cross the population.
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

    # Function to select the next generation.
    def Selection(self, population, cross_population, population_fitness, crossover_fitness):
        for i in range(len(population_fitness)):
            if crossover_fitness[i] >= population_fitness[i]:
                population[i] = cross_population[i]

class MP:
    """
    Class MP (Mapping Process) converts genotypes into phenotypes by replacing non-terminal symbols with terminal symbols.
    """

    # Constructor.
    def __init__(self):
        # Production rules.
        self.productions = {
            '<Start>': ['<Expr>'],
            '<Expr>': ['<Expr><Op><Expr>', '<Filter>(<Expr>)', '<Terminal>'],
            '<Filter>': ['<Gau>', '<Arith>', '<Lap>'],
            '<Gau>': ['ft.Gau1', 'ft.Gau2', 'ft.GauDX', 'ft.GauDY'],
            '<Lap>': ['ft.LapG1', 'ft.LapG2', 'ft.Lap'],
            '<Arith>': ['ft.Sqrt', 'ft.Sqr','ft.Log', 'ft.M05', 'ft.Abs', 'ft.Average', 'ft.Median', 'ft.HEq'],
            '<Op>': ['+', '-', '*', '/'],
            '<Terminal>': ['img']
        }

    # Function to expand a non-terminal symbol using the genotype.
    def expand_symbol(self, symbol, genotype, gen_index):
        if symbol in self.productions:
            choices = self.productions[symbol]
            choice_index = genotype[gen_index] % len(choices)
            return choices[int(choice_index)], gen_index + 1
        return symbol, gen_index


    # Main function to generate a string from the start symbol.
    def generate(self, genotype, wr):
        current_string = "<Start>"
        gen_index = 0
        if wr > 1:
            genotype * wr

        # Iterative expansion.
        while True:
            # Search for all non-terminals in the current string.
            non_terminals = re.findall(r'<[^>]+>', current_string)
            if any([gen_index == (len(genotype) - 1),  current_string == "img", current_string == "img-img", current_string == "img+img"]):
                return self.generate(np.random.randint(1,255,50), 3)            
            if not non_terminals:
                return current_string

            # Replace the first non-terminal found.
            for non_terminal in non_terminals:
                expansion, gen_index = self.expand_symbol(non_terminal, genotype, gen_index)
                current_string = current_string.replace(non_terminal, expansion, 1)
                break

gen = Genotype(50, 50, 1, 255, 0.5, 0.7)
mp = MP()

population = gen.init_population()
filters = [mp.generate(population[i], 3) for i in range(len(population))]

print(filters)