import unittest

import numpy as np
import matplotlib.pyplot as plt

from mdlib.forcefield import ForceField
from mdlib.bead_type import BeadType
from mdlib.potentials import LJWCA


def _compute_force_energy(
        r, eps=1.0, sigma=1.0, cut=2.5,
        lambda_lj=1.0, lambda_wca=0.0
):
    # create force field
    ff = ForceField()
    if not BeadType.has_bead_type("A"):
        BeadType("A")

    # create LJ potential and add to force field
    lj_potential = LJWCA()
    lj_potential.add_interaction(
        "A", "A", eps=eps, sigma=sigma, cut=cut,
        lambda_lj=lambda_lj, lambda_wca=lambda_wca
    )
    ff.add_potential(lj_potential)
    lj_potential.initialize()

    # initialize force and energy
    force = np.zeros_like(r)
    energy = np.zeros_like(r)

    # compute force and energy
    for i, _x in enumerate(r):
        r_i = np.zeros(3)
        r_i[0] = _x
        f_i, u_i = lj_potential.force_energy_function(np.zeros((2, 3)), 0.0, r_i, _x**2, 0, 1, 0, 0)
        force[i] = f_i[0, 0]
        energy[i] = u_i

    return force, energy


class TestPotential(unittest.TestCase):

    def test_lj(self):
        # plot
        r = np.linspace(0.95, 3.0, 1000)
        plt.figure("force")
        plt.axhline(linestyle='--', c='k')
        plt.figure("energy")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            force, energy = _compute_force_energy(r, eps=eps)
            plt.figure("force")
            plt.plot(r, force, label=f"{eps}")
            plt.figure("energy")
            plt.plot(r, energy, label=f"{eps}")
        for fig in ["force", "energy"]:
            plt.figure(fig)
            plt.legend(frameon=False, title="eps")
            if fig == "force":
                plt.ylabel(r"$|f_{ij}(r)|$")
                plt.savefig("test_lj_force.png")
            else:
                plt.ylabel(r"$u_{LJ}(r)$")
                plt.savefig("test_lj_energy.png")
        plt.close('all')

    def test_wca(self):
        # plot
        r = np.linspace(0.95, 3.0, 1000)
        plt.figure("force")
        plt.axhline(linestyle='--', c='k')
        plt.figure("energy")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            force, energy = _compute_force_energy(r, eps=eps, lambda_lj=0.0, lambda_wca=1.0)
            plt.figure("force")
            plt.plot(r, force, label=f"{eps}")
            plt.figure("energy")
            plt.plot(r, energy, label=f"{eps}")
        for fig in ["force", "energy"]:
            plt.figure(fig)
            plt.legend(frameon=False, title="eps")
            if fig == "force":
                plt.ylabel(r"$|f_{ij}(r)|$")
                plt.savefig("test_wca_force.png")
            else:
                plt.ylabel(r"$u_{WCA}(r)$")
                plt.savefig("test_wca_energy.png")
        plt.close('all')

    def test_lj_plus_wca(self):
        # plot
        r = np.linspace(0.95, 3.0, 1000)
        plt.figure("force")
        plt.axhline(linestyle='--', c='k')
        plt.figure("energy")
        plt.axhline(linestyle='--', c='k')
        for lambda_lj in [0.0, 0.25, 0.5, 0.75, 1.0]:
            force, energy = _compute_force_energy(r, lambda_lj=lambda_lj, lambda_wca=1.0-lambda_lj)
            plt.figure("force")
            plt.plot(r, force, label=f"{lambda_lj}")
            plt.figure("energy")
            plt.plot(r, energy, label=f"{lambda_lj}")
        for fig in ["force", "energy"]:
            plt.figure(fig)
            plt.legend(frameon=False, title=r"$\lambda_{lj}$")
            if fig == "force":
                plt.ylabel(r"$|f_{ij}(r)|$")
                plt.savefig("test_lj+wca_force.png")
            else:
                plt.ylabel(r"$u_{LJ}(r)$")
                plt.savefig("test_lj+wca_energy.png")
        plt.close('all')

    def test_plot_lj_plus_wca(self):
        r = np.linspace(0.95, 3.0, 1000)
        _, lj = _compute_force_energy(r, eps=1.0, sigma=1.0, cut=2.5, lambda_lj=1.0, lambda_wca=0.0)
        _, wca = _compute_force_energy(r, eps=1.0, sigma=1.0, cut=2.5, lambda_lj=0.0, lambda_wca=1.0)
        lj_plus_wca = lj + wca
        plt.figure()
        plt.axhline(y=0, linestyle='--', c='k')
        plt.plot(r, lj, label="LJ")
        plt.plot(r, wca, label="WCA")
        plt.plot(r, lj_plus_wca, label="LJ+WCA")
        plt.xlabel(r"$r / \sigma$")
        plt.ylabel(r"$u / \epsilon$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
