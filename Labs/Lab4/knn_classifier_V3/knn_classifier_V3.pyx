import numpy as np
cimport cython

DTYPE = np.intc
DTYPE_FLOAT = np.double


@cython.boundscheck(False)
@cython.wraparound(False)
def knn_classification(double[:, :] x_train, double[:] class_train, double[:, :] x_test, int K):  

    cdef Py_ssize_t N_test = x_test.shape[0]  
    cdef Py_ssize_t N_train = x_train.shape[0] 
    cdef Py_ssize_t D = x_train.shape[1]
    

    distances = np.zeros(N_train, dtype=DTYPE_FLOAT)
    cdef double[:] distances_view = distances

    class_pred = np.zeros(N_test, dtype=DTYPE_FLOAT)
    cdef double[:] class_pred_view = class_pred

    nearest_neighbors = np.zeros(K, dtype=DTYPE)
    cdef int[:] nearest_neighbors_view = nearest_neighbors

    nearest_labels = np.zeros(K, dtype=DTYPE_FLOAT)
    cdef double[:] nearest_labels_view = nearest_labels


    cdef Py_ssize_t q, i, d, k
    cdef double dist

    for q in range(N_test):
        
        for i in range(N_train):
            dist = 0.0
            for d in range(D):
                dist += (x_train[i, d] - x_test[q, d]) ** 2
            distances_view[i] = dist ** 0.5

        nearest_neighbors[:] = np.argsort(distances)[:K]

        for k in range(K):
            nearest_labels_view[k] = class_train[nearest_neighbors_view[k]]

        class_pred_view[q] = np.bincount(nearest_labels.astype(DTYPE)).argmax()

    return class_pred    
