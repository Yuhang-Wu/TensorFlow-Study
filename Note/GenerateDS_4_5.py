import numpy as np
import matplotlib.pyplot as plt

seed = 2


def generateds():
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300, 2)
    Y = [int(x1 * x1 + x2 + x2 < 2) for (x1, x2) in X]
    Y_color = [['red' if y else 'blue'] for y in Y]

    X = np.vstack(X).reshape(-1, 2)
    Y = np.vstack(Y).reshape(-1, 1)

    return X, Y, Y_color
