
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from joblib import Parallel, delayed
import random

PACE = 0.5
C = 30
THRESHOLD = 10
ERROR = 1 / 4
MUT_PROB = 0.33


def generate_traces(i, k, traces, mutation, n):
    traces[i, :n - k] = x[:n - k]
    if mutation:
        for j in range(n - k):
            if random.random() <= MUT_PROB:
                traces[i, j] = 1 - traces[i, j]
    traces[i, n - k:] = np.random.randint(2, size=k)
    return traces


def mean_based(x, n, T, mutation=False):
    x_pred = []
    start = 0
    end = n // 2
    threshold = int(30 * np.log2(n))


    traces = np.zeros((T, n), dtype=int)
    k_values = np.random.randint(n + 1, size=T)
    for i, k in enumerate(k_values):
        traces = generate_traces(i, k, traces, mutation, n)


    while (end < n) or (start != n):
        if n - start <= threshold:
            end = n


        mean_y_x = np.mean(traces[:,start:end], axis=0)
        x_pred.extend((mean_y_x >= 0.5).astype(int))

        if not mutation:
            mask = (x_pred[:end] != traces[:, :end]).any(axis=1)
            traces[mask, :] = 0
        else:
            for i in range(traces.shape[0]):
                t = traces[i, :end]
                # Find the absolute difference between the two arrays
                diff = np.abs(t - x_pred[:end])
                # Count the number of non-zero elements in the difference array
                diff_bits = np.count_nonzero(diff)
                if diff_bits >= (MUT_PROB+ERROR) * len(x_pred[start:end]):
                    np.delete(traces, i, axis=0)
        start = end
        end = min(int((end + n) * 0.5), n)

    errors = sum([1 if x[i] != x_pred[i] else 0 for i in range(n)]) / n
    print(errors)
    return errors


if __name__ == '__main__':
    corrects = 0
    n_s = []
    avg_error_n = []
    target_y = []
    target_x = []

    n = 64
    for _ in range(n, 1024, n * 2):
        x = [random.choice([0, 1]) for _ in range(n)]
        T = round(C * n * np.log2(n) ** 2)
        results = Parallel(n_jobs=14)(delayed(mean_based)(x, n, T, True) for _ in range(300))
        error = sum(results) / 300
        n_s.append(n)
        avg_error_n.append(error)
        print(f"{n} error: {error}")
        n *= 2


    for i in range(64, 1025):
        target_x.append(i)
        target_y.append(1 / i)


    plt.plot(n_s, avg_error_n, marker="o")
    plt.plot(target_x, target_y)
    plt.title("Error Estimation With Mutations")
    plt.legend(["Estimated Average Error", "Worst Case Error"])
    plt.xlabel("Input Length (n)")
    plt.ylabel("Probability Error (\u03B5)")
    plt.savefig(f"TrimmSuffixMutation")
    plt.show()
