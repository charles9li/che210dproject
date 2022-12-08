import matplotlib.pyplot as plt
import numpy as np


def get_free_energy_profile(eps_wall=None, sequence=None, file_prefix=None):
    if file_prefix is None:
        file_prefix = f"epswall{eps_wall}_{sequence}"
    bin_center_i = np.loadtxt(file_prefix+"_bin_center_i.txt")
    center_f_i = np.loadtxt(file_prefix+"_center_f_i.txt")
    center_df_i = np.loadtxt(file_prefix+"_center_df_i.txt")
    return bin_center_i, center_f_i, center_df_i


plt.figure("PMF")
plt.axhline(y=0, linestyle='--', c='k')
for color_index, sequence in enumerate(["ABABABABAB", "AAABBBBBAA", "AAAAABBBBB"]):
    color = f"C{color_index}"

    # get data
    x, f, df = get_free_energy_profile(eps_wall=1.0, sequence=sequence)

    # shift such that bulk free energy is 0
    shift = np.mean(f[x > 10])
    f = f - shift

    # calculate error in bulk free energy
    df_bulk = np.sqrt(np.sum(df[x > 10]**2)) / np.sum(x > 10)

    # find energy minimum, which is equivalent to the free energy of adsorption
    # since the bulk was shifted to zero
    min_f = np.min(f)

    # get the error in the energy minimum
    min_df = df[np.argmin(f)]

    # calculate error in energy of adsorption
    df_ads = np.sqrt(df_bulk**2 + min_df**2)

    # print values
    print(min_f, df_ads)

    # plot curve and error
    plt.plot(x, f, label=sequence, c=color)
    plt.fill_between(x, f - df, f + df, alpha=0.2)
    plt.plot([2.0, 12.0+color_index*3], [min_f, min_f], linestyle='--', c=color)

plt.legend(frameon=False, title="sequence")
plt.xlabel(r"$x / \sigma$")
plt.ylabel(r"$\beta F(x) - \beta F_{bulk}$")
plt.xlim((1.45, 20.0))
plt.show()
