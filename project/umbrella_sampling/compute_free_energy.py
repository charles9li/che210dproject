import os
from ast import literal_eval
from collections import defaultdict

import numpy as np

import pymbar

from utils import get_parameters_from_run_dir, create_bins


def compute_free_energy(
        eps_wall, sequence,
        min_bin_center=1.5, max_bin_center=10.0, bin_width=0.1,
        file_prefix=None, save_output=True
):
    # keep track of N_max
    N_max = 0

    # get umbrellas
    umbrellas = defaultdict(lambda: np.array([]))
    eps_wall_dir = f"epswall{eps_wall}"
    sequence_dir = os.path.join("data", eps_wall_dir, sequence)
    for k_bias_dir in os.listdir(sequence_dir):
        if k_bias_dir.startswith("kbias"):
            for x_dir in os.listdir(os.path.join(sequence_dir, k_bias_dir)):
                if x_dir.startswith("x"):
                    run_dir = os.path.join(sequence_dir, k_bias_dir, x_dir)
                    _, _, k_bias, x0 = get_parameters_from_run_dir(run_dir)
                    for x_samples_file in os.listdir(run_dir):
                        if x_samples_file.startswith("xsamples"):
                            x_samples_path = os.path.join(run_dir, x_samples_file)
                            print(f"Reading data in from {x_samples_path} ...")
                            x_n = np.loadtxt(x_samples_path)
                            key = (k_bias, x0)
                            umbrellas[key] = np.append(umbrellas[key], x_n)
                            N_max = max(N_max, len(umbrellas[key]))

    # allocate storage for simulation data
    K = len(umbrellas)  # number of umbrellas
    N_k = np.zeros([K], dtype=int)  # number of snapshots from umbrella simulation k
    K_k = np.zeros([K])             # spring constant for umbrella simulation k
    x0_k = np.zeros([K])            # bias center location for umbrella simulation k
    x_kn = np.zeros([K, N_max])     # distance from surface for snapshot n in umbrella simulation k
    u_kn = np.zeros([K, N_max])     # potential energy without umbrella restrains of snapshot n of umbrella simulation k

    # read in data
    for k, ((k_bias, x0), x_n) in enumerate(umbrellas.items()):
        N_k[k] = len(x_n)
        K_k[k] = k_bias
        x0_k[k] = x0
        x_kn[k, :N_k[k]] = x_n

    # u_kln[k, l, n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
    u_kln = np.zeros([K, K, N_max])

    # compute bin centers
    bin_center_i, bin_edges = create_bins(min_bin_center, max_bin_center, bin_width)

    N = np.sum(N_k)     # total count of all uncorrelated samples from all umbrellas
    x_n = pymbar.utils.kn_to_n(x_kn, N_k=N_k)

    # evaluate energies in all umbrellas
    print("Evaluating potential energies...")
    for k in range(K):
        for n in range(N_k[k]):
            dx = x_kn[k, n] - x0_k
            # compute energy of snapshot n from simulation in umbrella potential l
            u_kln[k, :, n] = (K_k / 2.0) * dx**2

    # initialize free energy profile with the data collected
    fes = pymbar.FES(u_kln, N_k, verbose=True)
    # compute free energy profile
    histogram_parameters = dict()
    histogram_parameters["bin_edges"] = bin_edges
    fes.generate_fes(u_kn, x_n, fes_type="histogram", histogram_parameters=histogram_parameters)
    results = fes.get_fes(bin_center_i, reference_point="from-lowest", uncertainty_method="analytical")
    center_f_i = results["f_i"]
    center_df_i = results["df_i"]

    # save outputs to file
    if save_output:
        if file_prefix is None:
            file_prefix = f"epswall{eps_wall}_{sequence}"
        np.savetxt(file_prefix+"_bin_center_i.txt", bin_center_i)
        np.savetxt(file_prefix+"_center_f_i.txt", center_f_i)
        np.savetxt(file_prefix+"_center_df_i.txt", center_df_i)

    return bin_center_i, center_f_i, center_df_i


if __name__ == '__main__':
    # for bin_width in [0.1, 0.2, 0.5, 1.0]:
    sequence = "AAABBBBBAA"
    for bin_width in [0.1]:
        compute_free_energy(1.0, sequence, bin_width=bin_width, max_bin_center=10.0)
