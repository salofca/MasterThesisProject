from collections import Counter

import numpy as np

"""
Auxiliary function which finds the most common element on an array
"""
def most_common_array(arrays):
    count = Counter([tuple(array) for array in arrays])
    most_common = count.most_common(1)
    return most_common[0][0]


n = 256
np.random.seed(123)
x = np.random.randint(2, size=n)
T = round(n * np.log2(n))
x_guess = []
traces = []

"""
Trace generation
"""
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
result = np.asarray(most_common_array(traces))
# Output the result
print("Most common binary string:", result)
print("The original string is: ", x)
print(x == result)
