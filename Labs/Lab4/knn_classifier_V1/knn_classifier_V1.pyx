
from cython cimport floating
import numpy as np
cimport numpy as np
from numpy cimport array, import_array

ctypedef fused myfloating:
    np.float32_t
    np.float64_t

ctypedef fused myint:
    np.int32_t
    np.int64_t

def knn_classification(myfloating[:, :] x_train, myint[:] class_train, myfloating[:, :] x_test, int K):  

    cdef Py_ssize_t N_test = x_test.shape[0]  
    cdef Py_ssize_t N_train = x_train.shape[0] 
    cdef Py_ssize_t D = x_train.shape[1]
    
    cdef myfloating[:] distances = np.zeros(N_train, dtype=x_train.dtype)
    cdef myint[:] class_pred = np.zeros(N_test, dtype=class_train.dtype)

    cdef myint[:] nearest_neighbors = np.zeros(K, dtype=np.int32)
    cdef myfloating[:] nearest_labels = np.zeros(K, dtype=class_train.dtype)

    cdef Py_ssize_t q, i, d, k
    cdef myfloating dist

    import_array()

    for q in range(N_test):
        
        for i in range(N_train):
            dist = 0
            for d in range(D):
                dist += (x_train[i, d] - x_test[q, d]) ** 2
            distances[i] = dist ** 0.5

        nearest_neighbors[:] = np.argsort(distances)[:K]

        for k in range(K):
            nearest_labels[k] = class_train[nearest_neighbors[k]]

        class_pred[q] = np.bincount(nearest_labels).argmax()

    return class_pred
