import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_percentage_error as MAPE


import IPython.display as display

from time import time

sns.set_style("darkgrid")

import tensorly as tl
import pyts.decomposition as pytsd

import argparse


def read_data(path, target_name):
    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        data = pd.read_excel(path)

    data['target'] = data[target_name]

    return data


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


def __main__():
    parser = argparse.ArgumentParser(
        description="Run the experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data file",
    )

    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Name of the target column',
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=24,
        help="Window size for the sliding window of hankelization and ssa",
    )

    parser.add_argument(
        "--parafac_rank",
        type=int,
        default=24,
        help="Rank for the parafac decomposition",
    )

    args = parser.parse_args()

    data = read_data(args.data, args.target)

    x = data['target'].values
    t = np.arange(0, len(x))

    window_width = args.window_size

    ssa = pytsd.SingularSpectrumAnalysis(window_size=window_width)
    ssa_x = ssa.fit_transform(x.reshape(1, -1))

    metrics_ssa = {
        'MSE': [],
        'MAPE': []
    }

    for k in range(0, window_width):
        first_k_sum = np.sum(ssa_x[:k+1], axis=0)

        mse = MSE(x, first_k_sum)
        mape = MAPE(x, first_k_sum)

        metrics_ssa['MSE'].append(mse)
        metrics_ssa['MAPE'].append(mape)

    plt.figure()
    sns.lineplot(x=np.arange(1, window_width + 1), y=metrics_ssa['MAPE'])
    plt.ylabel('MAPE')
    plt.xlabel('k-first components')
    plt.title('MAPE for SSA, window width = 24')
    plt.savefig('ssa_mape.png')

    hankel_x = hankel(hankel(x, window_width), 7)

    parafac_rank = args.parafac_rank
    parafac_x = tl.decomposition.parafac(
        hankel_x,
        rank=parafac_rank,

        n_iter_max=100,
        tol=1e-6,

        verbose=False
    )

    metricts_parafac = {
        'MSE': [],
        'MAPE': []
    }

    for i in range(1, parafac_rank):
        weights = np.zeros(parafac_rank)
        weights[:i] = 1
        parafac_x.weights = weights

        px = tl.cp_to_tensor(parafac_x)
        restored_x = unhankel(unhankel(px))

        mse = MSE(x, restored_x)
        mape = MAPE(x, restored_x)

        metricts_parafac['MSE'].append(mse)
        metricts_parafac['MAPE'].append(mape)

    plt.figure()
    sns.lineplot(x=np.arange(1, parafac_rank), y=metricts_parafac['MAPE'])
    plt.ylabel('MAPE')
    plt.xlabel('k-first components of khatri-rao')
    plt.title('MAPE for PARAFAC, window width = 24, rank = 24')
    plt.savefig('parafac_mape.png')

    plt.figure()
    sns.lineplot(x=np.arange(1, parafac_rank), y=metricts_parafac['MAPE'], label='PARAFAC')
    sns.lineplot(x=np.arange(1, window_width + 1), y=metrics_ssa['MAPE'], label='SSA')
    plt.ylabel('MAPE')
    plt.xlabel('k-first components')
    plt.title('Window width = 24')
    plt.legend()
    plt.savefig('comparison_mape.png')

if __name__ == "__main__":
    __main__()
