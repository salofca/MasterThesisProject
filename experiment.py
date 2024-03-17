import numpy as np
from joblib import Parallel, delayed
from numpy.random import rand
import pygad
from random import randint
from matplotlib import pyplot as plt


# average of population, must global in order to update it


# Fitness function
def fitness_function(ga, solution, ix_solution):
    return sum([1 if solution[i] == avg_pop[i] else 0 for i in range(len(solution))]) / len(solution)


# generate population
def generate_population(x, T, e):
    pop = []
    k_values = np.random.randint(len(x) + 1, size=T)
    for k in k_values:
        if k != 0:
            individual = x[:-k]
            for i in range(len(individual)):
                if rand() < e:
                    individual[i] = 1 - individual[i]
            suff = [randint(0, 1) for _ in range(k)]
            individual.extend(suff)
            pop.append(individual)
        else:
            individual = x.copy()
            for i in range(len(x)):
                if rand() < e:
                    individual[i] = 1 - individual[i]
            pop.append(individual)
    return pop


def callback_function(ga):
    verbose = 5
    last_fitness = 0
    atual_pop = ga.population
    avg_pop = np.mean(atual_pop, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    if ga.generations_completed % verbose == 0:
        print("Generation = {generation}".format(generation=ga.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga.best_solution()[1]))


# genetic algorithm

def genetic_algorithm(x, T):
    initial_population = generate_population(x, T, e)
    global avg_pop

    avg_pop = np.mean(initial_population, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    ga = pygad.GA(num_generations=50,
                  initial_population=initial_population,
                  fitness_func=fitness_function,
                  on_generation=callback_function,
                  num_parents_mating=1000,
                  parent_selection_type="tournament",
                  K_tournament=50,
                  sol_per_pop=2000,
                  crossover_type="two_points",
                  crossover_probability=0.9,
                  mutation_type="random",
                  mutation_probability=0.10,
                  gene_type=int,
                  )


    avg_pop = np.mean(ga.population, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    return sum([1 if x[i] != avg_pop[i] else 0 for i in range(len(x))]) / len(x)


# Let's start the algorithm
if __name__ == '__main__':
    n_s = []
    errors = []
    target_x = []
    target_y = []
    n = 256
    for _ in range(n, 1028, n * 2):
        for e in np.arange(0.1, 0.40, 0.1):
            error = 0
            x = [randint(0, 1) for _ in range(n)]
            T = int(40 * n * np.log2(n))
            for rep in range(100):
                error += genetic_algorithm(x, T)
            error = error / 100
            print(f"The average error for n = {n} is {error}")
            n_s.append(n)
            errors.append(error)
            n *= 2

    for i in range(64, 512):
        target_x.append(i)
        target_y.append(1 / i)

    plt.plot(n_s, errors, marker="o")
    plt.plot(target_x, target_y)
    plt.title("Error estimation With Mutations")
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig(f"GeneticAlgorithme033300generationsT10nlogn")
    plt.show()
