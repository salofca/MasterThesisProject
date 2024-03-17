# genetic algorithm search of the one max optimization problem
import numpy as np
from numpy.random import randint
from numpy.random import rand
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


# objective function
def min_error(ind, avg_pop):
    error = [1 if ind[i] != avg_pop[i] else 0 for i in range(len(ind))]
    return sum(error) / len(error)


# tournament selection
def selection(pop, scores, k=100):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def generate_population(x, n_pop):
    pop = []
    k_values = np.random.randint(len(x) + 1, size=n_pop)
    for k in k_values:
        if k != 0:
            individual = x[:-k]
            for i in range(len(individual)):
                if rand() < MUT_RATE:
                    individual[i] = 1 - individual[i]
            suff = [randint(0, 2) for _ in range(k)]
            individual.extend(suff)
            pop.append(individual)
        else:
            individual = x.copy()
            for i in range(len(x)):
                if rand() < MUT_RATE:
                    individual[i] = 1 - individual[i]
            pop.append(individual)
    return pop


def calculate_avg(pop):
    # Transpose the list of lists
    mean_y_x = np.mean(pop, axis=0)
    mean_y_x = (mean_y_x >= 0.5).astype(int)

    return mean_y_x


def genetic_algorithm(x, objective, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = generate_population(x, n_pop)

    # Calculate the average of the population
    avg_pop = calculate_avg(pop)
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c, avg_pop) for c in pop]
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        avg_pop = calculate_avg(pop)
        result = sum([1 if avg_pop[i] != x[i] else 0 for i in range(len(x))]) / len(x)
    result = sum([1 if avg_pop[i] != x[i] else 0 for i in range(len(x))]) / len(x)
    print(f"individual error {result}")
    return result


# define the total iterations
n_iter = 1
# bits
n_bits = 1024
# define the population size
n_pop = int(30 * n_bits * np.log2(n_bits))
# crossover rate
r_cross = 1
# mutation rate for algorithm
r_mut = 0.10

# mutation rate for population
MUT_RATE = 0.33

target_y = []
target_x = []
errors = []
n_s = []
n = 64
while n != 512:
    x = [randint(0, 2) for _ in range(n)]
    error = Parallel(n_jobs=-1)(
            delayed(genetic_algorithm)(x,min_error,n_iter,n_pop,r_cross,r_mut) for _ in range(300))
    error = sum(error) / 300
    print(f"average error {error}")
    n = n*2
    errors.append(error)
    n_s.append(n)


for i in range(64, 512):
    target_x.append(i)
    target_y.append(1 / i)

plt.plot(n_s, errors, marker="o")
plt.plot(target_x, target_y)
plt.title("Error estimation With Mutations")
plt.legend(["Estimated Average Error", "Worst Case Error"])
plt.xlabel("Input Length (n)")
plt.ylabel("Probability Error (\u03B5)")
plt.savefig(f"GeneticAlgorithm30nlogntraces")
plt.show()
