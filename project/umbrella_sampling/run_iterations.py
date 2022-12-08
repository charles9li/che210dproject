import multiprocessing as mp

import numpy as np

from compute_free_energy import compute_free_energy
from distributions import get_overall_distribution, create_bins
from run_umbrella import run_umbrella


def compute_mse(errors):
    return np.sqrt(np.mean(errors ** 2))


# parameters
min_bin_center = 1.5
max_bin_center = 20.0
bin_width = 0.1
default_umbrellas = np.arange(min_bin_center, max_bin_center+bin_width, bin_width, dtype=float)
sequence = "AAAAABBBBB"
eps_wall = 1.0
k_bias = 5.0
steps_per_simulation = 10000000
n_processors = 4
abs_tol = 1e-2

# first check to see if any default umbrella centers are unsampled
all_x = get_overall_distribution(eps_wall, sequence)
bin_centers, bin_edges = create_bins(1.5, 20.0, 0.1)
hist, bin_edges = np.histogram(all_x, bins=bin_edges)
umbrellas_to_sample = []
for u in default_umbrellas:
    if hist[np.argwhere(bin_centers == u)[0][0]] == 0:
        print(f"No samples detected at {u} ...")
        umbrellas_to_sample.append(u)
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

while current_mse > abs_tol:
    # get distributions to see where to find where to sample
    all_x = get_overall_distribution(eps_wall, sequence)
    bin_centers, bin_edges = create_bins(1.5, 20.0, 0.1)
    hist, bin_edges = np.histogram(all_x, bins=bin_edges)
    indices = np.argpartition(hist, n_processors)[:n_processors]
    umbrellas_to_run = bin_centers[indices]
    print(f"Least sampled bin centers at {' '.join([str(u) for u in umbrellas_to_run])}")

    # run all the simulations
    print("Running simulations with umbrellas at this bin centers ...")
    pool = mp.Pool(n_processors)
    for u in umbrellas_to_run:
        kwargs = {'steps': steps_per_simulation, 'verbose': False, 'keep_traj': False}
        pool.apply_async(run_umbrella, args=(eps_wall, sequence, k_bias, u), kwds=kwargs)
    pool.close()
    pool.join()

    # compute the current MSE in free energy
    _, _, center_df_i = compute_free_energy(eps_wall, sequence,
                                            min_bin_center=min_bin_center, max_bin_center=max_bin_center)
    current_mse = compute_mse(center_df_i)
    print(f"Current MSE of {current_mse} after this iteration ...")
    print(" ")
