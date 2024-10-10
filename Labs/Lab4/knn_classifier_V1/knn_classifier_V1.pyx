import numpy as np

DTYPE = np.intc
DTYPE_FLOAT = np.double


def knn_classification(x_train, class_train, x_test, int K):  

    cdef Py_ssize_t N_test = x_test.shape[0]  
    cdef Py_ssize_t N_train = x_train.shape[0] 
    cdef Py_ssize_t D = x_train.shape[1]
    
    distances = np.zeros(N_train, dtype=DTYPE_FLOAT)
    class_pred = np.zeros(N_test, dtype=DTYPE_FLOAT)
    nearest_neighbors = np.zeros(K, dtype=DTYPE)
    nearest_labels = np.zeros(K, dtype=DTYPE_FLOAT)

    cdef Py_ssize_t q, i, d, k
    cdef double dist

    for q in range(N_test):
        
        for i in range(N_train):
            dist = 0.0
            for d in range(D):
                dist += (x_train[i, d] - x_test[q, d]) ** 2
            distances[i] = dist ** 0.5

        nearest_neighbors[:] = np.argsort(distances)[:K]

        for k in range(K):
            nearest_labels[k] = class_train[nearest_neighbors[k]]

        class_pred[q] = np.bincount(nearest_labels.astype(DTYPE)).argmax()

    return class_pred    
