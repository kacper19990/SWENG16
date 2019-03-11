import numpy as np
import os


def get(dataset):
    root = os.path.dirname(os.path.abspath(__file__))
    path = root + "/" + dataset + ".npz"

    f = np.load(path)
    x_train = f['arr_0.npy']
    x_test = f['arr_1.npy']
    f.close()

    return x_train, x_test
