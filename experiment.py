from array import array
import random

from deap_er import algorithms, creator, base, tools
import numpy as np

N = 512
x = np.random.randint(2, size=N)
T = int(2 * N * np.log2(N))
traces = []
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array, typecode='i', fitness=creator.FitnessMin)


def evaluate(individual, ancestor):
    result = [1 if individual[i] != ancestor[i] else 0 for i in range(N)]
    return sum(result) / N


toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(N), N)

toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.init_repeat, list, toolbox.individual)

toolbox.register("mate", tools.cx_two_point)
toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
toolbox.register("select", tools.sel_tournament, contestants=10)
toolbox.register("evaluate", evaluate, ancestor=x)


def main():
    pop = toolbox.population(size=T)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.ea_simple(population=pop, toolbox=toolbox, cx_prob=0.25, mut_prob=0.3, generations=500, stats=stats,
                         hof=hof,verbose=True)

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
    print(stats.fields)
    print(hof)
    print(evaluate(hof[0], x))
