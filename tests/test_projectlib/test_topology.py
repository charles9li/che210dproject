import unittest

from projectlib.bead_type import BeadType
from projectlib.topology import Bead, LinearChain, Topology


class TestTopology(unittest.TestCase):

    def test_homopolymer_melt(self):
        # create the bead type
        bead_type_a = BeadType("A")

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


if __name__ == '__main__':
    unittest.main()
