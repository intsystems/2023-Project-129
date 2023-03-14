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

    print(args)


if __name__ == "__main__":
    __main__()
