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


"""
Calculates the error between the predicted string and the original string
"""


def calculate_error(x, x_pred):
    return sum([1 if x[i] != x_pred[i] else 0 for i in range(x.shape[0])]) / x.shape[0]


n_s = []
avg_error = []
target_y = []
target_x = []
for n in range(64, 1028, 128):
    error = 0
    for _ in range(300):
        x = np.random.randint(2, size=n)
        T = int(n * np.log2(n))
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
        error += calculate_error(x, x_pred)
    print(f"Error for n = {n} : {error / 300}  ")
    avg_error.append(error / 300)
    n_s.append(n)

for i in range(64, 1025):
    target_x.append(i)
    target_y.append(1 / i)

plt.plot(n_s, avg_error, marker="o")
plt.plot(target_x, target_y)
plt.title("Error estimation Trimmsuffix")
plt.legend(["Estimated Average Error", "Worst Case Error"])
plt.xlabel("Input Length (n)")
plt.ylabel("Probability Error (\u03B5)")
plt.savefig(f"TrimSuffixTnlogn")
plt.show()
