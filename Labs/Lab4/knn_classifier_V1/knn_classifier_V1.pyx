import numpy as np
cimport numpy as np


DTYPE_FLOAT = np.float
DTYPE_INT = np.int_c

def knn_classification(x_train, class_train, x_test, int K):

    cdef Py_ssize_t N_test = x_test.shape[0]
    cdef Py_ssize_t N_train = x_train.shape[0]

    distances = np.zeros(N_train, dtype=DTYPE_FLOAT)
    class_pred = np.zeros(N_test, dtype=DTYPE_INT)

    cdef Py_ssize_t q, i


    for q in range(N_test):
        distances = np.linalg.norm(x_train - x_test[q, np.newaxis], axis=1)

        nearest_neighbors = np.argsort(distances)[:K]
        nearest_labels = class_train[nearest_neighbors]

        class_pred[q] = np.bincount(nearest_labels).argmax()

    return class_pred
