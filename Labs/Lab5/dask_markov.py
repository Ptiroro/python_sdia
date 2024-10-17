import numpy as np
from multiprocess_markov import markov
from dask.distributed import Client, as_completed


def dask_markov(rho, A, nmax, seed):
    rng = np.random.default_rng(seed =seed)
    return markov(rho, A, nmax, rng)


if __name__ == "__main__":
    client = Client()

    A = np.array([[0.3, 0.4, 0.3],
                  [0.4, 0.3, 0.3],
                  [0.3, 0.3, 0.4]])
    rho = np.array([0.3, 0.3, 0.4])
    nmax = 10

    n_simul = 30

    futures = []
    for seed in range(n_simul):
        future = client.submit(dask_markov, rho, A, nmax, seed)
        futures.append(future)

    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        print(f"Simulation {i+1}: {result}")

    client.close()
