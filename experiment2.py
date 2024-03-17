import random
from deap_er import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np

# (Define the problem and individual classes as before)
N = range(5, 21, 5)
EPSILON = 0.25


# Define the functions to create individuals and populations
def create_individual(length):
    return np.array([random.choice([0, 1]) for _ in range(length)])


def mean_based_ancestor(population):
    mean_y_x = np.mean(population, axis=1)
    return mean_y_x



def create_population(individual_creator, population_size, individual_length):
    traces = np.zeros((population_size, individual_length))
    for i in range(population_size):  # Trace generation
        k = int(np.random.uniform(0, individual_length))
        if k != 0:
            traces[i, :k - individual_length] = individual_creator[:-k]
            for j in range(individual_length - k):
                if random.random() <= EPSILON:
                    traces[i, j] = 1 - traces[i, j]
    traces = creator.Individual(traces)
    return traces


# Define the evaluation function for trace reconstruction
def evaluate(individual, ancestor):
    result = [1 if individual[i] != ancestor[i] else 0 for i in range(len(ancestor))]
    return sum(result) / len(ancestor)


# Create the toolbox


def run_algorithm(ancestor_length, ancestor):
    population_size = int(20 * np.log2(ancestor_length))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, length=ancestor_length)
    toolbox.register("population", create_population, individual_creator=ancestor,
                     population_size=population_size, individual_length=ancestor_length)
    toolbox.register("evaluate", evaluate, ancestor=ancestor)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.1)
    toolbox.register("select", tools.sel_tournament, contestants=2)
    # Create the population
    population = toolbox.population()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)
    # Crossover and mutate for 10 generations
    result, log = algorithms.ea_simple(population=population, toolbox=toolbox, cx_prob=0.4, mut_prob=0.2,
                                       generations=100, stats=stats, hof=hof, verbose=True)

    # Return the best individual in the final population

    return tools.sel_best(population, sel_count=1)[0].fitness.values[0]


def main():
    # Set the range of input lengths (n)
    # Change this range as needed
    errors = []
    n_s = []
    for n in range(128, 2048, 256):
        ancestor = list(np.random.randint(2, size=n))
        print(f"ancestor: {ancestor}")
        error = run_algorithm(n, ancestor)
        n_s.append(n)
        errors.append(error)

    # Plot the results
    plt.plot(n_s, errors, marker='o')
    plt.xlabel('Input Length (n)')
    plt.ylabel('Error')
    plt.title('Error vs. Input Length')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
