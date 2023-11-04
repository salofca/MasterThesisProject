from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

"""
Auxiliary function which finds the most common element on an array
"""


def most_common_array(arrays):
    count = Counter([tuple(array) for array in arrays])
    most_common = count.most_common(1)
    return most_common[0][0]


def are_traces_equal(x, x_pred):
    for (xi, gi) in zip(x, x_pred):
        if xi != gi:
            return 0
    return 1


correct = 0
n_s = []
avg_error = []
for n in range(256, 2048, 64):
    x = np.random.randint(2, size=n)
    T = int(1.2 * n * np.log2(n))
    x_pred = []
    traces = []
    for i in range(T):
        k = int(np.random.uniform(0, n))
        if k != 0:
            trace = x[:-k]
            suffix = np.random.randint(2, size=k)
            trace = np.concatenate((trace, suffix), axis=None)
            traces.append(trace)
        else:
            traces.append(x)

    # Find the most common binary string
    x_pred = np.asarray(most_common_array(traces))
    # Output the result
    correct += are_traces_equal(x, x_pred)
    n_s.append(n)
    avg_error.append((1 - (correct / len(n_s))))

plt.plot(n_s, avg_error)
plt.xlabel("Input Length (n)")
plt.ylabel("Average Error (\u03B5)")
plt.show()
