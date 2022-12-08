import matplotlib.pyplot as plt
import numpy as np


def get_free_energy_profile(eps_wall=None, sequence=None, file_prefix=None):
    if file_prefix is None:
        file_prefix = f"epswall{eps_wall}_{sequence}"
    bin_center_i = np.loadtxt(file_prefix+"_bin_center_i.txt")
    center_f_i = np.loadtxt(file_prefix+"_center_f_i.txt")
    center_df_i = np.loadtxt(file_prefix+"_center_df_i.txt")
    return bin_center_i, center_f_i, center_df_i


plt.figure()
for bin_width in [0.1]:
    x, f, df = get_free_energy_profile(eps_wall=1.0, sequence="AAAAABBBBB")
    plt.errorbar(x, f, yerr=df)
plt.show()
