import time

import numpy as np
from matplotlib import pyplot as plt
import random

PACE = 0.50
C = 2
THRESHOLD = 100
MIN_DIFF = 0.7
ERROR = 20


def mean_based(x, n, T, mutation=False):
    x_pred = []
    traces = []
    start = 0
    end = n // 2
    remaining_traces = 0
    epsilon = 0
    seen_part = 0
    threshold = 64
    rows_delete = []

    traces = np.zeros((T, n), dtype=int)
    while (end < n) or (start != n):
        k_values = np.random.randint(n + 1, size=T - remaining_traces)
        if len(rows_delete) > 0:
            for (ix, k) in zip(rows_delete, k_values):
                traces[ix, :n - k] = x[:n - k]
                if mutation:
                    epsilon = np.random.uniform(0, 1, size=len(traces))
                    traces = 1 - traces * (epsilon >= 0.5)
                traces[ix, n - k:] = np.random.randint(2, size=k)

        else:
            for i, k in enumerate(k_values):
                traces[i, :n - k] = x[:n - k]
                if mutation:
                    epsilon = np.random.uniform(0, 1, size=len(traces))
                    traces = 1 - traces * (epsilon >= 0.5)
                traces[i, n - k:] = np.random.randint(2, size=k)

        # stacking the traces all together

        if n - start <= threshold:
            end = n

        mean_y_x = np.mean(traces[:, start:end], axis=0)
        x_pred.extend((mean_y_x >= 0.5).astype(int))

        # deleting the traces which their bits are different from x

        rows_delete = []
        if not mutation:
            for i in range(traces.shape[0]):
                t = traces[i, :end]
                for (xi, ti) in zip(x_pred[:end], t):
                    if xi != ti:
                        rows_delete.append(i)
                        traces[i, :] = 0
                        break
        else:
            for i in range(traces.shape[0]):
                t = traces[i, :end]
                # Find the absolute difference between the two arrays
                diff = np.abs(t - x[:end])
                # Count the number of non-zero elements in the difference array
                diff_bits = np.count_nonzero(diff)
                if diff_bits <= MIN_DIFF * len(x[:end]) + ERROR:
                    rows_delete.append(i)

        remaining_traces = traces.shape[0] - len(rows_delete)
        seen_part = end - start
        start = end
        end = min(end + int(PACE * end), n)
        threshold += seen_part + (seen_part * np.sqrt(seen_part))

    errors = np.mean(x != np.array(x_pred))
    return errors


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    target_y = []
    target_x = []

    for n in range(64,1100 , 128):  # Applying mean-based over a sequence of n
        corrects = 0
        error = 0
        for rep in range(1, 500):
            x = [random.choice([0, 1]) for _ in range(n)]
            T = round(2 * n * np.log2(n))
            result = mean_based(x, n, T, False)
            error += result
        error = error / 500
        n_s.append(n)
        avg_error_n.append(error)
        print(f"{n} error: {error}")

    for i in range(64, 512):
        target_x.append(i)
        target_y.append(1 / i)

    i = 0
    plt.plot(n_s, avg_error_n, marker="o")
    plt.plot(target_x, target_y)
    plt.title("Error estimation")
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig("MeanBasedPartitionsAvgError " + str(i + 1))
    plt.show()
