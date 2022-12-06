import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md

from pymbar import timeseries


plt.figure()
for k_bias in [0.1, 0.5, 1.0, 5.0, 10.0]:
    # get data
    t = md.load(f"k{k_bias}_traj.pdb")
    x = np.mean(t.xyz[:, :, 0], axis=1)

    # get equilibrated and uncorrelated samples
    t0, g, _ = timeseries.detect_equilibration(x)
    x_equil = x[t0:]
    indices = timeseries.subsample_correlated_data(x_equil, g=g)
    x_n = x_equil[indices]

    # find optimal number of bins
    q25, q75 = np.percentile(x_n, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x_n) ** (-1/3)
    bins = round((np.max(x_n) - np.min(x_n)) / bin_width)

    # plot
    plt.hist(x_n, density=True, bins=bins, alpha=0.5, label=f"{k_bias}")

plt.legend(title="k_bias", frameon=False)
plt.show()
