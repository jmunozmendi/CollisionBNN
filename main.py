import pandas as pd
import numpy as np

from collision_bnn import CBNN, CBNNInfo


def main():
    x = pd.read_csv('data/x.csv').to_numpy()
    y = pd.read_csv('data/y.csv').to_numpy().flatten()

    x_train = x[:8000, :]
    y_train = y[:8000]

    x_test = x[8000:, :]
    y_test = y[8000:]

    collision_checker = CBNN(12)

    collision_checker.train(x_train, y_train)

    info = collision_checker.predict(x_test)

    result = np.logical_and(info.decision, y_test)

    print(np.sum(result)/len(result))


if __name__ == '__main__':
    main()
