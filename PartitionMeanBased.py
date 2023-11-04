import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

PACE = 0.25
C = 3


def mean_based(n, T):
    x_pred = []
    traces = []
    start = 0
    end = n // 2
    remain_traces = 0

    while (end < n) or (start != n):
        traces = list(traces)
        for i in range(T - remain_traces):
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
                x_pred.append(1)
            else:
                x_pred.append(0)

        # deleting the traces which their bits are different from x
        rows_delete = []
        for i in range(traces.shape[0]):
            t = traces[i, :end]
            for (xi, ti) in zip(x[:end], t):
                if xi != ti:
                    rows_delete.append(i)
                    break

        traces = np.delete(traces, rows_delete, 0)
        remain_traces = traces.shape[0]
        start = end
        end = min(end + int(PACE * end), n)

    for (xi, gi) in zip(x, x_pred):
        if xi != gi:
            return 0

    return 1


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    for n in range(256, 2048, 64):
        x = np.random.randint(2, size=n)
        T = int(C * n * np.log2(n))
        result = mean_based(n, T)
        corrects += result
        n_s.append(n)
        avg_error_n.append(1 - (corrects / len(n_s)))
        print((1 - (corrects / len(n_s))))

    plt.plot(n_s, avg_error_n)
    plt.xlabel("Input Length (n)")
    plt.ylabel("Average Error (\u03B5)")
    plt.savefig("MeanBasedPartitionsAvgError")
    plt.show()
