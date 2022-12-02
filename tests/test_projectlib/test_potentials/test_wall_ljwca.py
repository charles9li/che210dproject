import unittest

import numpy as np
import matplotlib.pyplot as plt

from projectlib import BeadType, ForceField, WallLJWCA


def _compute_force_energy(
        x, eps=1.0, sigma=1.0, cut=7.5,
        axis=0, lower_bound=0.0, upper_bound=1.0,
        lambda_lj=1.0, lambda_wca=0.0
):
    # create force field
    ff = ForceField()
    bead_type = BeadType("A")
    ff.add_bead_type(bead_type)

    # create LJ potential and add to force field
    wall_lj_potential = WallLJWCA()
    wall_lj_potential.add_interaction(
        "A", eps=eps, sigma=sigma, cut=cut,
        axis=axis, lower_bound=lower_bound, upper_bound=upper_bound,
        lambda_lj=lambda_lj, lambda_wca=lambda_wca
    )
    ff.add_potential(wall_lj_potential)

    # initialize force and energy
    force = np.zeros_like(x)
    energy = np.zeros_like(x)

    # compute force and energy
    for i, _x in enumerate(x):
        r_i = np.zeros(3)
        r_i[axis] = _x
        f_i, u_i = wall_lj_potential.force_energy_function(np.zeros((1, 3)), 0.0, r_i, 0, 0)
        force[i] = f_i[0, axis]
        energy[i] = u_i

    return force, energy


class TestWallLJWCA(unittest.TestCase):

    def test_wall_lj(self):
        lower_bound = 0.0
        upper_bound = 20.0
        x = np.linspace(lower_bound + 0.995, upper_bound - 0.995, 1000)
        plt.figure("force")
        plt.axhline(linestyle='--', c='k')
        plt.figure("energy")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            force, energy = _compute_force_energy(x, eps=eps, lower_bound=lower_bound, upper_bound=upper_bound)
            plt.figure("force")
            plt.plot(x, force, label=f"{eps}")
            plt.figure("energy")
            plt.plot(x, energy, label=f"{eps}")
        for i, fig in enumerate(["force", "energy"]):
            plt.figure(fig)
            plt.legend(frameon=False, title="eps")
            plt.xlabel(r"$x / \sigma$")
            if i == 0:
                plt.ylabel(r"$f_x$")
            else:
                plt.ylabel(r"$u_{wall}$")
            if i == 0:
                plt.savefig("test_wall_lj_force.png")
            else:
                plt.savefig('test_wall_lj_energy.png')

    def test_wall_wca(self):
        lower_bound = 0.0
        upper_bound = 5.0
        x = np.linspace(lower_bound + 0.995, upper_bound - 0.995, 10000)
        plt.figure("force")
        plt.axhline(linestyle='--', c='k')
        plt.figure("energy")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            force, energy = _compute_force_energy(x, eps=eps, lower_bound=lower_bound, upper_bound=upper_bound, lambda_wca=1.0, lambda_lj=0.0)
            plt.figure("force")
            plt.plot(x, force, label=f"{eps}")
            plt.figure("energy")
            plt.plot(x, energy, label=f"{eps}")
        for i, fig in enumerate(["force", "energy"]):
            plt.figure(fig)
            plt.legend(frameon=False, title="eps")
            plt.xlabel(r"$x / \sigma$")
            if i == 0:
                plt.ylabel(r"$f_x$")
            else:
                plt.ylabel(r"$u_{wall}$")
            if i == 0:
                plt.savefig("test_wall_wca_force.png")
            else:
                plt.savefig('test_wall_wca_energy.png')
        plt.show()


if __name__ == '__main__':
    unittest.main()
