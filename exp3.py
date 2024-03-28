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
def generate_population(x, T):
    pop = []
    k_values = np.random.randint(len(x), size=T)

    for k in k_values:
        if k != 0:
            l = np.random.randint(len(x) - k)
            individual = x[l:-k]
            suff = [randint(0, 1) for _ in range(k)]
            pref = [randint(0, 1) for _ in range(l)]
            pref.extend(individual)
            pref.extend(suff)
            pop.append(pref)
        else:
            individual = x.copy()
            l = np.random.randint(len(x) + 1)
            individual = individual[l:]
            pref = [ randint(0,1) for _  in range(l)]
            pref.extend(individual)
            pop.append(pref)
    return pop


def callback_function(ga):
    verbose = 2
    atual_pop = ga.population
    avg_pop = np.mean(atual_pop, axis=0)
    avg_pop = (avg_pop >= 0.5).astype(int)

    if ga.generations_completed % verbose == 0:
        print("Generation = {generation}".format(generation=ga.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga.best_solution()[1]))


# genetic algorithm

def genetic_algorithm(x, T):
    initial_population = generate_population(x, T)
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
    i = 0
    while n < 1025:
            j = 0
            x = [randint(0, 1) for _ in range(n)]
            T = int(20*n * np.log2(n))
            error = Parallel(n_jobs=10)(
                delayed(genetic_algorithm)(x, T) for _ in range(24
                                                                   ))
            error = sum(error) / 24
            print(f"The average error for n = {n}  is {error}")
            #errors[i, j] = e
            #j += 1
        #i += 1
            n_s.append(n)
            errors.append(error)
            n*=2

    for i in range(64, 1025):
        target_x.append(i)
        target_y.append(1 / i)

    plt.plot(n_s, errors, marker="o")
    plt.plot(target_x, target_y)
    plt.title(f"Error estimation T=20nlogn")
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig(f"GeneticAlgorithmn20logntrimextend")
    plt.show()





