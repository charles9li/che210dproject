import math
import time
import multiprocessing as mp

import numpy as np

from compute_free_energy import compute_free_energy
from distributions import get_overall_distribution, create_bins
from run_umbrella import run_umbrella


def compute_mse(errors):
    return np.sqrt(np.mean(errors ** 2))


# parameters
min_bin_center = 2.0
max_bin_center = 20.0
bin_width = 0.1
default_umbrellas = np.arange(2, 21, step=2, dtype=float)
sequence = "AAABBBBBAA"
eps_wall = 1.0
k_bias = 5.0
steps_per_simulation = 10000000
n_processors = 4
abs_tol = 1e-2

# first check to see if any default umbrella centers are unsampled
all_x = get_overall_distribution(eps_wall, sequence)
max_x = np.max(all_x)
bin_centers, bin_edges = create_bins(min_bin_center, 20.0, 0.1)
hist, bin_edges = np.histogram(all_x, bins=bin_edges)
umbrellas_to_sample = []
for u in default_umbrellas:
    if u > max_x:
        print(f"No samples detected at {u} ...")
        umbrellas_to_sample.append(u)
umbrellas_to_sample = np.around(umbrellas_to_sample, decimals=math.ceil(math.log10(1 / bin_width)))
if len(umbrellas_to_sample) > 0:
    print("Running simulations at default umbrellas ...")
    pool = mp.Pool(n_processors)
    for u in umbrellas_to_sample:
        kwargs = {'steps': steps_per_simulation, 'verbose': False, 'keep_traj': False}
        pool.apply_async(run_umbrella, args=(eps_wall, sequence, k_bias, u), kwds=kwargs)
    pool.close()
    pool.join()

# compute the current MSE in free energy
_, _, center_df_i = compute_free_energy(eps_wall, sequence,
                                        min_bin_center=1.5, max_bin_center=20.0)
current_mse = compute_mse(center_df_i)
print(f"Initial MSE of {current_mse} ...")

while current_mse > abs_tol:
    # get distributions to see where to find where to sample
    all_x = get_overall_distribution(eps_wall, sequence)
    bin_centers, bin_edges = create_bins(min_bin_center, max_bin_center, 0.1)
    hist, bin_edges = np.histogram(all_x, bins=bin_edges)
    indices = np.argpartition(hist, n_processors)[:n_processors]
    umbrellas_to_run = np.around(bin_centers[indices], decimals=math.ceil(math.log10(1 / bin_width)))
    print(f"Least sampled bin centers at {' '.join([str(np.around(u, decimals=1)) for u in umbrellas_to_run])}")

    # run all the simulations
    start_time = time.time()
    print("Running simulations with umbrellas at this bin centers ...")
    pool = mp.Pool(n_processors)
    for u in umbrellas_to_run:
        kwargs = {'steps': steps_per_simulation, 'verbose': False, 'keep_traj': False}
        pool.apply_async(run_umbrella, args=(eps_wall, sequence, k_bias, u), kwds=kwargs)
    pool.close()
    pool.join()
    print(f"Simulations finished in {time.time() - start_time} s ...")

    # compute the current MSE in free energy
    _, _, center_df_i = compute_free_energy(eps_wall, sequence,
                                            min_bin_center=1.5, max_bin_center=20.0)
    current_mse = compute_mse(center_df_i)
    print(f"Current MSE of {current_mse} after this iteration ...")
    print(" ")
