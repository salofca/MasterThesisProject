import numpy as np
from matplotlib import pyplot as plt


PACE = 0.50
C = 2
THRESHOLD = 100
MIN_DIFF = 0.7
ERROR = 20




def mean_based(n, T, mutation=False):
    x_pred = []
    traces = []
    start = 0
    end = n // 2
    remaining_traces = 0
    epsilon = 0

    while (end < n) or (start != n):
        traces = list(traces)
        for i in range(T - remaining_traces):  # Trace generation
            k = np.random.randint(n+1)
            if k != 0:
                trace = x[:-k].copy()
                if mutation:
                    epsilon = np.random.uniform(0,1)
                    for ix,ti in enumerate(trace):
                        if epsilon >= 0.5:
                            trace[ix] = 1-ti
                suffix = np.random.randint(2, size=k)
                trace = np.concatenate((trace, suffix), axis=None)
                traces.append(trace)
            else:
                traces.append(x.copy())

        traces = np.vstack(traces)  # stacking the traces all together

        if n - start <= THRESHOLD:
            end = n

        mean_y_x = np.mean(traces[:, start:end], axis=0)
        x_pred.extend((mean_y_x >= 0.5).astype(int))

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
                if diff_bits <= MIN_DIFF*len(x[:end]) + ERROR:
                    rows_delete.append(i)

        traces = np.delete(traces, rows_delete, 0)
        remaining_traces = traces.shape[0]
        start = end
        end = min(end + int(PACE * end), n)
    errors = np.mean(x != np.array(x_pred))
    return errors


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    target_y = []
    target_x = []

    for n in range(128, 512, 64):  # Applying mean-based over a sequence of n
        corrects = 0
        error = 0
        for rep in range(1, 500):
            x = np.random.randint(2, size=n)
            T = round(10 * n * np.log2(n))
            result = mean_based(n, T,True)
            error += result
            print(f"rep: {rep}")
        error = error / 500
        n_s.append(n)
        avg_error_n.append(error)
        print(f"{n} error: {error}")

    for i in range(64, 512):
        target_x.append(i)
        target_y.append(1 / i)

    i = 0
    plt.plot(n_s, avg_error_n,marker="o")
    plt.plot(target_x, target_y)
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig("MeanBasedPartitionsAvgError " + str(i + 1))
    plt.show()
