import os
from pathlib import Path
from ast import literal_eval

import numpy as np

import mdtraj as md


def create_run_dir(eps_wall, sequence, k_bias, x0):
    run_dir = os.path.join("data", f"epswall{eps_wall}", sequence, f"kbias{k_bias}", f"x{x0}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_parameters_from_run_dir(run_dir):
    path_split = os.path.normpath(run_dir).split(os.sep)
    eps_wall = literal_eval(path_split[1][7:])
    sequence = path_split[2]
    k_bias = literal_eval(path_split[3][5:])
    x0 = literal_eval(path_split[4][1:])
    return eps_wall, sequence, k_bias, x0


def load_free_energy_profile_from_files(eps_wall=None, sequence=None, file_prefix=None):
    if file_prefix is None:
        file_prefix = f"epswall{eps_wall}_{sequence}"
    bin_center_i = np.loadtxt(file_prefix+"_bin_center_i.txt")
    center_f_i = np.loadtxt(file_prefix+"_center_f_i.txt")
    center_df_i = np.loadtxt(file_prefix+"_center_df_i.txt")
    return bin_center_i, center_f_i, center_df_i


def find_uncorrelated_samples(traj_file):
    from pymbar import timeseries
    t = md.load(traj_file)
    x = np.mean(t.xyz[:, :, 0], axis=1)
    t0, g, _ = timeseries.detect_equilibration(x)
    x_t_equil = x[t0:]
    indices = timeseries.subsample_correlated_data(x_t_equil, g=g)
    x_n = x_t_equil[indices]
    return x_n


def create_bins(min_bin_center, max_bin_center, bin_width=0.1):
    n_bins = round((max_bin_center - min_bin_center) / bin_width) + 1
    bin_center_i = min_bin_center + bin_width * np.arange(n_bins)
    bin_edges = min_bin_center - bin_width / 2.0 + bin_width * np.arange(n_bins+1)
    return bin_center_i, bin_edges
