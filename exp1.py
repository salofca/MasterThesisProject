import numpy as np
from joblib import Parallel, delayed
from numpy.random import rand
import pygad
from random import randint
from matplotlib import pyplot as plt

# average of population, must global in order to update it

MUT_RATE = 0.33


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
    verbose = 2
    last_fitness = 0
    atual_pop = ga.population
    avg_pop = np.mean(atual_pop, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    if ga.generations_completed % verbose == 0:
        print("Generation = {generation}".format(generation=ga.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga.best_solution()[1]))


# genetic algorithm

def genetic_algorithm(x, T, e):
    initial_population = generate_population(x, T, e)
    global avg_pop

    avg_pop = np.mean(initial_population, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    ga = pygad.GA(num_generations=20,
                  initial_population=initial_population,
                  fitness_func=fitness_function,
                  on_generation=callback_function,
                  num_parents_mating=T // 300,
                  parent_selection_type="tournament",
                  K_tournament=20,
                  crossover_type="scattered",
                  crossover_probability=0.20,
                  mutation_type="random",
                  mutation_probability=0.00,
                  gene_type=int,
                  )

    ga.run()
    avg_pop = np.mean(ga.population, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)
    error = sum([1 if x[i] != avg_pop[i] else 0 for i in range(len(x))]) / len(x)
    print(f"individual error {error}")
    return error


# Let's start the algorithm
if __name__ == '__main__':

    n_s = []
    errors = []
    target_x = []
    target_y = []

    n = 64
    for _ in range(n, 512, n * 2):
        for e in np.arange(0.1, 0.5, 0.1):
            x = [randint(0, 1) for _ in range(n)]
            T = int(10 / (0.5 - e) * n * np.log2(n))
            error = Parallel(n_jobs=-1)(
                delayed(genetic_algorithm)(x, T, e) for _ in range(24
                                                                   ))
            error = sum(error) / 24
            print(f"The average error for n = {n} and error = {e} is {error}")
            n_s.append(n)
            errors.append(error)
        n *= 2

    for i in range(64, 1025):
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
