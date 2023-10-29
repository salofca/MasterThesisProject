import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn import linear_model


def mean_based(c):
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

        # deleting the traces which their length is different from x
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
        end = min(end + int(c * end), n)

    corrects = 0
    for (xi, gi) in zip(x, x_guess):
        if xi == gi:
            corrects += 1

    return corrects


if __name__ == '__main__':
    n = 256
    np.random.seed(123)
    x = np.random.randint(2, size=n)
    T = round(np.log2(n))
    corrects = []
    c_s = []

    for c in np.arange(0.0081,0.1,0.001):
        result = mean_based(c)
        corrects.append(result)
        c_s.append(c)
        print(f"c {c} result {result}")

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(np.array(c_s).reshape(-1,1), corrects)
    m = lin_reg.coef_
    b = lin_reg.intercept_

    plt.scatter(c_s,corrects)
    plt.xlabel("Pace value")
    plt.ylabel("Correct bit predictions")
    y = m*c_s+b
    plt.plot(c_s,y)
    plt.show()




