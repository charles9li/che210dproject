import matplotlib.pyplot as plt

from project.umbrella_sampling.utils import load_free_energy_profile_from_files


plt.figure()
for bin_width in [0.1, 0.2, 0.5, 1.0]:
    x, f, df = load_free_energy_profile_from_files(file_prefix=f"binwidth{bin_width}")
    plt.errorbar(x, f, yerr=df, label=f"{bin_width}")
plt.legend(frameon=False, title=r"bin width ($\sigma$)")
plt.xlabel(r"$x / \sigma$")
plt.ylabel(r"$F / kT$")
plt.show()
