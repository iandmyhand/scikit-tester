import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import logging
from core.logger import set_logger

logger = logging.getLogger('scikit-tester')


def get_data():
    iris_dataset = load_iris()
    return train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


def learn():
    X_train, X_test, y_train, y_test = get_data()
    logger.debug("X_train shape: {}".format(X_train.shape))
    logger.debug("y_train shape: {}".format(y_train.shape))
    logger.debug("X_test shape: {}".format(X_test.shape))
    logger.debug("y_test shape: {}".format(y_test.shape))

    # create dataframe from data in X_train
    # label the columns using the strings in iris_dataset.feature_names
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train
    grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


def learn2():
    X_train, X_test, y_train, y_test = get_data()

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)


if '__main__' == __name__:
    set_logger()
    learn2()
