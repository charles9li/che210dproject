import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md

from pymbar import timeseries


# parameters
reporting_frequencies = [100, 500, 1000, 5000, 10000]

# iterate through reporting frequencies
correlation_times = []
for reporting_frequency in reporting_frequencies:
    # get data
    t = md.load(f"rf{reporting_frequency}_traj.pdb")
    x = np.mean(t.xyz[:, :, 0], axis=1)

    # detect equilibration and compute correlation time
    t0, g, _ = timeseries.detect_equilibration(x)
    print(f"Set 0 reporting frequency = {reporting_frequency}")
    print(f"    t0 = {t0} and g = {g}")
    print(f"    correlation time = {g*reporting_frequency} steps")
    correlation_times.append(g*reporting_frequency)

plt.figure()
plt.plot(reporting_frequencies, correlation_times)
plt.xscale('log')
plt.show()
