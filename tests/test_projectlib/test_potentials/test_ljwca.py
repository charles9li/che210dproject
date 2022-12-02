import unittest

import numpy as np
import matplotlib.pyplot as plt

from projectlib import ForceField, BeadType, LJWCA


class TestPotential(unittest.TestCase):

    def test_lj(self):
        # create force field
        ff = ForceField()
        bead_type = BeadType("A")
        ff.add_bead_type(bead_type)

        # create LJ potential and add to force field
        lj_potential = LJWCA()
        lj_potential.add_interaction("A", "A")
        ff.add_potential(lj_potential)

        # plot
        r = np.linspace(0.95, 3.0, 1000)
        u = np.zeros_like(r)
        f = np.zeros_like(r)
        plt.figure("test_lj_energy")
        plt.axhline(linestyle='--', c='k')
        plt.figure("test_lj_force")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            lj_potential.set_parameter_value('eps', eps, "A", "A")
            lj_potential.create_force_energy_function()
            for i, _r in enumerate(r):
                r_ij = np.array([_r, 0, 0])
                forces, u[i] = lj_potential.force_energy_function(np.zeros((2, 3)), 0.0, r_ij, _r**2, 0, 1, 0, 0)
                f[i] = forces[0, 0]
            plt.figure("test_lj_energy")
            plt.plot(r, u, label=f"{eps}")
            plt.figure("test_lj_force")
            plt.plot(r, f, label=f"{eps}")
        for i, fig in enumerate(["test_lj_energy", "test_lj_force"]):
            plt.figure(fig)
            plt.xlabel("r")
            if i == 0:
                plt.ylabel(r"$u_{LJ}(r)$")
            else:
                plt.ylabel(r"$|f_{ij}(r)|$")
            plt.legend(frameon=False, title=r"$\epsilon$")
            plt.savefig(f"{fig}.png")

    def test_wca(self):
        # create force field
        ff = ForceField()
        bead_type = BeadType("A")
        ff.add_bead_type(bead_type)

        # create LJ potential and add to force field
        lj_potential = LJWCA()
        lj_potential.add_interaction("A", "A", lambda_lj=0.0, lambda_wca=1.0)
        ff.add_potential(lj_potential)

        # plot
        r = np.linspace(0.99, 1.5, 1000)
        u = np.zeros_like(r)
        f = np.zeros_like(r)
        plt.figure("test_wca_energy")
        plt.axhline(linestyle='--', c='k')
        plt.figure("test_wca_force")
        plt.axhline(linestyle='--', c='k')
        for eps in [0.5, 1.0, 1.5, 2.0]:
            lj_potential.set_parameter_value('eps', eps, "A", "A")
            lj_potential.create_force_energy_function()
            for i, _r in enumerate(r):
                r_ij = np.array([_r, 0, 0])
                forces, u[i] = lj_potential.force_energy_function(np.zeros((2, 3)), 0.0, r_ij, _r**2, 0, 1, 0, 0)
                f[i] = forces[0, 0]
            plt.figure("test_wca_energy")
            plt.plot(r, u, label=f"{eps}")
            plt.figure("test_wca_force")
            plt.plot(r, f, label=f"{eps}")
        for i, fig in enumerate(["test_wca_energy", "test_wca_force"]):
            plt.figure(fig)
            plt.xlabel("r")
            if i == 0:
                plt.ylabel(r"$u_{WCA}(r)$")
            else:
                plt.ylabel(r"$|f_{ij}(r)|$")
            plt.legend(frameon=False, title=r"$\epsilon$")
            plt.savefig(f"{fig}.png")

    def test_lj_plus_wca(self):
        # create force field
        ff = ForceField()
        bead_type = BeadType("A")
        ff.add_bead_type(bead_type)

        # create LJ potential and add to force field
        lj_potential = LJWCA()
        lj_potential.add_interaction("A", "A", lambda_lj=0.0, lambda_wca=1.0)
        ff.add_potential(lj_potential)

        # plot
        r = np.linspace(0.99, 3.0, 1000)
        u = np.zeros_like(r)
        f = np.zeros_like(r)
        plt.figure("test_lj+wca_energy")
        plt.axhline(linestyle='--', c='k')
        plt.figure("test_lj+wca_force")
        plt.axhline(linestyle='--', c='k')
        for lambda_lj in [0.0, 0.25, 0.50, 0.75, 1.0]:
            lj_potential.set_parameter_value('lambda_lj', lambda_lj, "A", "A")
            lj_potential.set_parameter_value('lambda_wca', 1-lambda_lj, "A", "A")
            lj_potential.create_force_energy_function()
            for i, _r in enumerate(r):
                r_ij = np.array([_r, 0, 0])
                forces, u[i] = lj_potential.force_energy_function(np.zeros((2, 3)), 0.0, r_ij, _r**2, 0, 1, 0, 0)
                f[i] = forces[0, 0]
            plt.figure("test_lj+wca_energy")
            plt.plot(r, u, label=f"{lambda_lj}")
            plt.figure("test_lj+wca_force")
            plt.plot(r, f, label=f"{lambda_lj}")
        for i, fig in enumerate(["test_lj+wca_energy", "test_lj+wca_force"]):
            plt.figure(fig)
            plt.xlabel("r")
            if i == 0:
                plt.ylabel(r"$u_{LJ+WCA}(r)$")
            else:
                plt.ylabel(r"$|f_{ij}(r)|$")
            plt.legend(frameon=False, title=r"$\lambda_{lj}$")
            plt.savefig(f"{fig}.png")


if __name__ == '__main__':
    unittest.main()
