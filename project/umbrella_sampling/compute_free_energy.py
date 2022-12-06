import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md

import pymbar
from pymbar import timeseries


# parameters
sequence = "AAAAABBBBB"
k_bias = 1.0
umbrellas = np.arange(2, 20, 2, dtype=float)
K = len(umbrellas)  # number of umbrellas
N_max = 1000        # maximum number of snapshots/simulation
x_min = 1.5         # min for distance from surface
x_max = 20.0        # max for distance from surface
n_bins = 100         # number of bins for 1D free energy profile

# allocate storage for simulation data
N_k = np.zeros([K], dtype=int)  # number of snapshots from umbrella simulation k
K_k = np.zeros([K])             # spring constant for umbrella simulation k
x0_k = np.zeros([K])            # bias center location for umbrella simulation k
x_kn = np.zeros([K, N_max])     # distance from surface for snapshot n in umbrella simulation k
u_kn = np.zeros([K, N_max])     # potential energy without umbrella restrains of snapshot n of umbrella simulation k
g_k = np.zeros([K])

for k in range(K):
    x0_k[k] = umbrellas[k]
    K_k[k] = k_bias

# read in data
for k, x0 in zip(range(K), umbrellas):
    filename = f"{sequence}_r{x0}_traj.pdb"
    print(f"Reading {filename}...")
    traj = md.load(filename)
    x_t = np.mean(traj.xyz[:, :, 0], axis=1)
    n = len(x_t)
    x_kn[k, :n] = x_t
    N_k[k] = n

    # detect equilibration period
    t0, g, _ = timeseries.detect_equilibration(x_t)
    x_t_equil = x_t[t0:]
    print(f"Warmup length for set    {k:5d} is {t0:10d}")
    print(f"Correlation time for set {k:5d} is {g:10.3f}")

    # get number of uncorrelated samples of x
    indices = timeseries.subsample_correlated_data(x_t_equil, g=g)
    x_n = x_t_equil[indices]

    # store data
    N_k[k] = len(indices)
    u_kn[k, 0:N_k[k]] = u_kn[k, indices]
    x_kn[k, 0:N_k[k]] = x_kn[k, indices]

# shorten the array size
N_max = np.max(N_k)
# u_kln[k, l, n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
u_kln = np.zeros([K, K, N_max])

# compute bin centers
bin_center_i = np.zeros([n_bins])
bin_edges = np.linspace(x_min, x_max, n_bins + 1)
for i in range(n_bins):
    bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i+1])

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

np.savetxt("bin_center_i.txt", bin_center_i)
np.savetxt("center_f_i.txt", center_f_i)
np.savetxt("center_df_i.txt", center_df_i)

plt.figure()
plt.errorbar(bin_center_i, center_f_i, yerr=center_df_i)
plt.show()
