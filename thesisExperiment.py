import numpy as np
import itertools

n = 256
np.random.seed(123)
x = np.random.randint(2, size=n)
T = round(np.log2(n))
x_guess = []
traces = []
start = 0
end = n//2

for i in range(T):
    k = int(np.random.uniform(0, n))
    if k != 0:
        trace = x[:-k]
        suffix = np.random.randint(2, size=k)
        trace = np.concatenate((trace, suffix), axis=None)
        traces.append(trace)
    else:
        traces.append(x)
traces = np.vstack(traces) # stacking the traces all together

for j in range(start, end):
    mean_y_x = np.mean(traces[:, j], axis=0)
    if mean_y_x >= 0.5:
        x_guess.append(1)
    else:
        x_guess.append(0)

for i in range(int(traces.shape[1]/2)): # verifying if the prediction is equal to the original
    print(x[i] == x_guess[i])

for i in range(traces.shape[0]):
    t = traces[i,end+1]

    






