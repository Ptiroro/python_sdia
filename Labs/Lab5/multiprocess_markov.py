# your code
import multiprocessing
import numpy as np

def markov(rho, A, nmax, rng):

    assert A.shape[0] == A.shape[1] # A matrice carré
    assert np.all(np.isclose(A.sum(axis = 1), 1)) # A matrice stochastique

    assert np.all(rho >= 0)
    assert rho.ndim == 1 and rho.size == A.shape[0]
    assert np.isclose(rho.sum(), 1)

    N = A.shape[0]
    X = np.zeros(nmax, dtype=int)

    X[0] = rng.choice(np.arange(N), p=rho) # Initialisation de X_0 avec la loi de rho

    q = 0
    while q < nmax - 1:
        current_state = X[q]

        T = rng.geometric(A[current_state, current_state]) - 1

        for t in range(min(T + 1, nmax - q - 1)):
            X[q + t + 1] = current_state

        q += T + 1

        # Determiner le nouvel état
        if q < nmax:
            next_state = rng.choice(np.arange(N), p=A[current_state])
            X[q] = next_state

    return X

def multiprocess_markov(params):
    rho, A, nmax, seed = params
    rng = np.random.default_rng(seed=seed)
    return markov(rho, A, nmax, rng)

if __name__ == "__main__":

    A = np.array([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3], [0.3, 0.3, 0.4]])
    rho = np.array([0.3, 0.3, 0.4])
    nmax = 10

    n_simul = 30

    params = [(rho, A, nmax, seed) for seed in range(0, n_simul)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(multiprocess_markov, params)

    print(f"A:{A}")
    print(f"rho:{rho}")
    print(f"nmax:{nmax}")
    print('-'*10 + 'Resultats' + '-'*10)
    for i, result in enumerate(results):
        print(f"Simulation {i+1}: {result}")
