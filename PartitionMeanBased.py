import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

PACE = 0.50
C = 10
THRESHOLD = 16
MIN_DIFF = 0.001
ERROR = 0.01

def mean_based(n, T, mutation=False):
    x_pred = []
    traces = []
    start = 0
    end = n // 2
    remaining_traces = 0

    while (end < n) or (start != n):
        traces = list(traces)
        for i in range(T - remaining_traces):  # Trace generation
            k = int(np.random.uniform(0, n))
            if k != 0:
                trace = x[:-k]
                if mutation:
                    epsilon = np.random.uniform(0,1)
                    for ix,ti in enumerate(trace):
                        if epsilon >= 0.5:
                            trace[ix] = 1-ti
                suffix = np.random.randint(2, size=k)
                trace = np.concatenate((trace, suffix), axis=None)
                traces.append(trace)
            else:
                traces.append(x)

        traces = np.vstack(traces)  # stacking the traces all together

        if n - start <= THRESHOLD:
            end = n

        mean_y_x = np.mean(traces[:, start:end], axis=0)
        for y_bar in mean_y_x:
            if y_bar >= 0.5:
                x_pred.append(1)
            else:
                x_pred.append(0)

        # deleting the traces which their bits are different from x

        rows_delete = []
        if not mutation:
            for i in range(traces.shape[0]):
                t = traces[i, :end]
                for (xi, ti) in zip(x[:end], t):
                    if xi != ti:
                        rows_delete.append(i)
                        break
        else:
            for i in range(traces.shape[0]):
                t = traces[i,:end]
                # Find the absolute difference between the two arrays
                diff = np.abs(t - x[:end])
                # Count the number of non-zero elements in the difference array
                diff_bits = np.count_nonzero(diff)
                if diff_bits >= MIN_DIFF*len(x[:end]) + ERROR:
                    rows_delete.append(i)

        traces = np.delete(traces, rows_delete, 0)
        remaining_traces = traces.shape[0]
        start = end
        end = min(end + int(PACE * end), n)
    errors = [1 if x[i] != x_pred[i] else 0 for i in range(len(x))]
    return sum(errors) / len(x)


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    target_y = []
    target_x = []
    repetions = 0
    for n in range(64, 256, 16):  # Applying mean-based over a sequence of n
        corrects = 0
        error = 0
        for rep in range(1, 1000):
            x = np.random.randint(2, size=n)
            T = round(C * n * np.log2(n))
            result = mean_based(n, T,True)
            error += result
            repetions += rep
        error = error / 1000
        n_s.append(n)
        avg_error_n.append(error)
        print(error)

    for i in range(256, 1000):
        target_x.append(i)
        target_y.append(1 / i)

    i = 0
    plt.plot(n_s, avg_error_n)
    plt.plot(target_x, target_y)
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig("MeanBasedPartitionsAvgError " + str(repetions) + str(i + 1))
    plt.show()
