import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils import get_parameters_from_run_dir, create_bins


def get_distribution_from_each_umbrella(eps_wall, sequence):
    umbrellas = defaultdict(lambda: np.array([]))
    eps_wall_dir = f"epswall{eps_wall}"
    sequence_dir = os.path.join("data", eps_wall_dir, sequence)
    for k_bias_dir in os.listdir(sequence_dir):
        if k_bias_dir.startswith("kbias"):
            for x_dir in os.listdir(os.path.join(sequence_dir, k_bias_dir)):
                if x_dir.startswith("x"):
                    run_dir = os.path.join(sequence_dir, k_bias_dir, x_dir)
                    _, _, k_bias, x0 = get_parameters_from_run_dir(run_dir)
                    for samples_file in os.listdir(run_dir):
                        if samples_file.startswith("xsamples"):
                            x_samples_path = os.path.join(run_dir, samples_file)
                            x_n = np.loadtxt(x_samples_path)
                            umbrellas[(k_bias, x0)] = np.append(umbrellas[(k_bias, x0)], x_n)
    return umbrellas


def get_overall_distribution(eps_wall, sequence):
    x_n = np.array([])
    for umbrella in get_distribution_from_each_umbrella(eps_wall, sequence).values():
        x_n = np.append(x_n, umbrella)
    return x_n


if __name__ == '__main__':
    # bin_centers, bin_edges = create_bins(1.5, 20.0, 0.1)
    # distributions = get_distribution_from_each_umbrella(1.0, "AB"*5)
    # plt.figure()
    # for dist in distributions.values():
    #     hist, bin_edges = np.histogram(dist, bins=bin_edges)
    #     plt.hist(bin_edges[:-1], bin_edges, alpha=0.3, weights=hist)

    bin_centers, bin_edges = create_bins(1.5, 20.0, 0.1)
    hist, bin_edges = np.histogram(get_overall_distribution(1.0, "AAAAABBBBB"), bins=bin_edges)
    plt.hist(bin_edges[:-1], bin_edges, weights=hist)

    plt.show()
