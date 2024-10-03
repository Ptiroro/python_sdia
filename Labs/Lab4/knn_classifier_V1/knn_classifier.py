import numpy as np


def knn_classification(x_train, class_train, x_test, K):
    N_test = x_test.shape[0]
    class_pred = np.zeros(N_test)

    for q in range(N_test):
        distances = np.linalg.norm(x_train - x_test[q, np.newaxis], axis=1)

        nearest_neighbors = np.argsort(distances)[:K]
        nearest_labels = class_train[nearest_neighbors]

        class_pred[q] = np.bincount(nearest_labels.astype(int)).argmax()

    return class_pred