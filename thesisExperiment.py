import numpy as np
import itertools

n = 256
np.random.seed(123)
x = np.random.randint(2, size=n)
T = round(np.log2(n))
x_guess = []
traces = []
start = 0
end = n // 2
del_traces = 0

while (end < n) or (start != n):
    traces = list(traces)
    for i in range(T - del_traces):
        k = int(np.random.uniform(0, n))
        if k != 0:
            trace = x[:-k]
            suffix = np.random.randint(2, size=k)
            trace = np.concatenate((trace, suffix), axis=None)
            traces.append(trace)
        else:
            traces.append(x)

    traces = np.vstack(traces)  # stacking the traces all together

    for j in range(start, end):
        mean_y_x = np.mean(traces[:, j], axis=0)
        if mean_y_x >= 0.5:
            x_guess.append(1)
        else:
            x_guess.append(0)

    # deleting the traces which their half-length is different from x
    rows_delete = []
    for i in range(traces.shape[0]):
        t = traces[i, :end]
        for (xi, ti) in zip(x[:end], t):
            if xi != ti:
                rows_delete.append(i)
                break

    traces = np.delete(traces, rows_delete, 0)
    del_traces = traces.shape[0]
    start = end
    end = min(end + int(0.008 * end), n)
    print(end)

print(x_guess[:] == x[:])
