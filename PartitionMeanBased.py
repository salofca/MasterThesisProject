import time

import numpy as np
from matplotlib import pyplot as plt
import random
import time
from joblib import Parallel, delayed

PACE = 0.5
C = 1000
THRESHOLD = 100
MIN_DIFF = 0.7
ERROR = 20
MUT_PROB = 0.25


def mean_based(x, n, T, mutation=False):
    x_pred = []
    traces = []
    start = 0
    end = n // 2
    remaining_traces = 0
    epsilon = 0
    seen_part = 0
    threshold = 0
    rows_delete = []

    traces = np.zeros((T, n), dtype=int)
    while (end < n) or (start != n):

        k_values = np.random.randint(n + 1, size=T - remaining_traces)
        if len(rows_delete) > 0:
            for (i, k) in zip(rows_delete, k_values):
                traces[i, :n - k] = x[:n - k]
                if mutation:
                    for j in range(n-k):
                        if random.random() <= MUT_PROB:
                            traces[i,j] = 1 - traces[i,j]
                traces[i, n - k:] = np.random.randint(2, size=k)

        else:
            for i, k in enumerate(k_values):
                traces[i, :n - k] = x[:n - k]
                if mutation:
                    for j in range(n - k):
                        if random.random() <= MUT_PROB:
                            traces[i, j] = 1 - traces[i, j]
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
        threshold = MUT_PROB * seen_part + C * np.sqrt(seen_part * np.log2(seen_part))

    errors = np.mean(x != np.array(x_pred))
    return errors


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    target_y = []
    target_x = []
    T = 0

    for n in range(64, 1024, 128):  # Applying mean-based over a sequence of n
        corrects = 0
        error = 0
        x = [random.choice([0, 1]) for _ in range(n)]
        T = round(5 * n * np.log2(n) ** 2)
        results = Parallel(n_jobs=-1)(delayed(mean_based)(x, n, T, True) for _ in range(500))
        error = sum(results) / 500
        print(results)
        n_s.append(n)
        avg_error_n.append(error)
        print(f"{n} error: {error}")

    for i in range(64, 1025):
        target_x.append(i)
        target_y.append(1 / i)

    i = 0
    plt.plot(n_s, avg_error_n, marker="o")
    plt.plot(target_x, target_y)
    plt.title("Error estimation With Mutations")
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig(f"TrimmSuffixMutation T = {T}")
    plt.show()
