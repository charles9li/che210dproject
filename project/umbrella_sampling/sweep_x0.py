import multiprocessing as mp

import numpy as np

from run_umbrella import run_umbrella


eps_wall = 1.0
sequences = ["AAAAABBBBB"]
# umbrellas = np.arange(11, 15, dtype=float) + 0.5
umbrellas = np.array([10.5, 16, 17, 18, 19, 20], dtype=float)

pool = mp.Pool(6)
for sequence in sequences:
    for x0 in umbrellas:
        pool.apply_async(run_umbrella, args=(eps_wall, sequence, 5.0, x0))
pool.close()
pool.join()
