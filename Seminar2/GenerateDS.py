import numpy as np


def GenerateInputs(gamma):
    """
    Sample two classes of points: where "gamma" is a margin parameter.
        a. 200 data points from a uniform distribution on [gamma/2, 1] x [-1/2, 1/2]
        b. 200 data points from a uniform distribution on [-1, -gamma/2] x [-1/2, 1/2]
    """
    np.random.seed(1)

    a1 = np.random.uniform(-1.0, -gamma / 2.0, size=200)
    a2 = np.random.uniform(-0.5, 0.5, size=200)

    x1 = np.random.uniform(gamma / 2.0, 1.0, size=200)
    x2 = np.random.uniform(-0.5, 0.5, size=200)

    c1 = np.squeeze(np.dstack((a1, a2)))
    c2 = np.squeeze(np.dstack((x1, x2)))

    # create labels
    y1 = np.negative(np.ones(200))
    y2 = np.ones(200)

    data = np.concatenate((c1, c2))
    labels = np.concatenate((y1, y2))

    # shuffle the data in the same way
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]

    return data, labels


def ScatterPlot(data, labels, gamma):
    """ Plot the input data points with corresponding labels """

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, label in enumerate(labels):
        plt.scatter(data[i, 0], data[i, 1], label=label,
                    color=['red' if label < 0 else 'green'])
    plt.axvline(gamma / 2, linestyle='--', color='indigo', alpha=0.3)
    plt.axvline(-gamma / 2, linestyle='--', color='indigo', alpha=0.3)
    plt.show()
