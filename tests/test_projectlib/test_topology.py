import unittest

import numpy as np

from mdlib.bead_type import BeadType
from mdlib.topology import Bead, LinearChain, Topology
from mdlib.potentials import LJWCA
from mdlib.forcefield import ForceField
from mdlib.system import System
from mdlib.integrators import LangevinIntegrator
from mdlib.simulation import Simulation


class TestTopology(unittest.TestCase):

    def test_homopolymer_melt(self):
        # create the bead type
        try:
            bead_type_a = BeadType("A")
        except ValueError:
            bead_type_a = BeadType.get_by_name("A")

        # create hompolymer of length 4 and check number of beads and bonds
        homopolymer = LinearChain("homopolymer", [Bead("A", bead_type_a) for _ in range(4)])
        self.assertEqual(4, homopolymer.n_beads)
        self.assertEqual(3, homopolymer.n_bonds)

        # create topology and add 5 hompolymer chains to it
        topology = Topology()
        topology.add_chain(homopolymer, n=5)

        # check the number of beads and bonds in the topology
        self.assertEqual(20, topology.n_beads)
        self.assertEqual(15, topology.n_bonds)

        # check that chain and bead indices are correct
        self.assertListEqual(list(range(5)), [chain.index for chain in topology.chains])
        self.assertListEqual(list(range(20)), [bead.index for bead in topology.beads])

    def test_chain_to_pdb(self):
        # create the bead type
        try:
            bead_type_a = BeadType("A")
        except ValueError:
            bead_type_a = BeadType.get_by_name("A")

        # create homopolymer of length 4
        hompolymer = LinearChain("homopolymer", [Bead("A", bead_type_a) for _ in range(4)])

        # save to pdb
        hompolymer.to_pdb("homopolymer.pdb")

    def test_lj_fluid_pdb(self):
        # parameters
        N = 240
        rho = 0.8
        L = (N / rho)**(1/3)

        # create the bead type
        try:
            bead_type_a = BeadType("A")
        except ValueError:
            bead_type_a = BeadType.get_by_name("A")

        # create LJ particle and topology
        lj_particle = LinearChain("lj", [Bead("A", bead_type_a)])
        topology = Topology()
        topology.add_chain(lj_particle, n=N)
        topology.box_lengths = np.array([L]*3)

        # create interactions and force field
        lj_potential = LJWCA()
        lj_potential.add_interaction("A", "A")
        force_field = ForceField()
        force_field.add_potential(lj_potential)

        # create system and integrator
        system = System(topology, force_field)
        integrator = LangevinIntegrator()

        # create simulation
        simulation = Simulation(system, integrator)
        simulation.initialize()
        topology.to_pdb("lj_fluid.pdb", positions=simulation.state.positions)


if __name__ == '__main__':
    unittest.main()
