import numpy as np


def get(dataset):
    # Load arrays from file
    f = np.load(dataset + ".npz")
    x_train = f['arr_0.npy']
    x_test = f['arr_1.npy']
    f.close()

    return x_train, x_test
