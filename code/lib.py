import numpy as np


def hankel(x, window_size):
    return np.array([x[i:i+window_size] for i in range(0, len(x) - window_size + 1)])

def unhankel(x):
    dims_left = list(x.shape[2:])
    shape = [x.shape[0] + x.shape[1] - 1] + dims_left

    sums = np.zeros(shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sums[i+j] += x[i,j]

    for i in range(len(sums)):
        d = min([i + 1, x.shape[0], x.shape[1], len(sums) - i])
        sums[i] /= d

    return sums