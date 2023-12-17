# genetic algorithm search of the one max optimization problem
import numpy as np
from numpy.random import randint
from numpy.random import rand


# objective function
def min_error(ind,avg_pop):
    error = [ 1 if ind[i] != avg_pop[i] else 0 for i in range(len(ind))]
    return sum(error) / len(error)


# tournament selection
def selection(pop, scores, k=3):
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
            pop.append(x)
    return pop


def calculate_avg(pop):
    transposed_lists = list(map(list, zip(*pop)))
    # Calculate the average for each coordinate
    average_list = [sum(bits) / len(bits) for bits in transposed_lists]
    average_list = [1 if xibar >= 0.5 else 0 for xibar in average_list]
    return average_list


def genetic_algorithm(x,objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring

    print(x)
    pop = generate_population(x, n_pop)

    # Calculate the average of the population
    avg_pop = calculate_avg(pop)
    # keep track of best solution
    best, best_eval = 0, objective(pop[0],avg_pop)
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c,avg_pop) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
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
    return [best, best_eval]


# define the total iterations
n_iter = 20
# bits
n_bits = 2048
# define the population size
n_pop = int(n_bits * np.log2(n_bits))
# crossover rate
r_cross = 0.4
# mutation rate for algorithm
r_mut = 1.0 / float(n_bits)

# mutation rate for population
MUT_RATE = 0.25
# perform the genetic algorithm search
x = [randint(0, 2) for _ in range(n_bits)]
best, score = genetic_algorithm(x,min_error, n_bits, n_iter, n_pop, r_cross, r_mut)
print(x == best)
print('Done!')
print('f(%s) = %f' % (best, score))

