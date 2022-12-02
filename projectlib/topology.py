"""topology.py: Used for storing topological information about a system."""
__all__ = ['Bead', 'Chain', 'LinearChain', 'Topology']

from copy import deepcopy

from .bead_type import BeadType


class Bead(object):

    def __init__(self, name, bead_type):
        self.name = name
        if isinstance(bead_type, str):
            self.bead_type = BeadType.get_by_name(bead_type)
        else:
            self.bead_type = bead_type
        self.index_in_chain = None
        self.index = None


class Chain(object):

    def __init__(self, name, beads):
        self.name = name
        self._beads = []
        for i, bead in enumerate(beads):
            bead.index_in_chain = i
            self._beads.append(bead)
        self._bonds = []
        self.index = None

    def add_bond(self, bead1, bead2):
        self._bonds.append((bead1, bead2))

    @property
    def beads(self):
        return iter(self._beads)

    @property
    def bonds(self):
        return iter(self._bonds)

    @property
    def n_beads(self):
        return len(self._beads)

    @property
    def n_bonds(self):
        return len(self._bonds)


class LinearChain(Chain):

    def __init__(self, name, beads):
        super(LinearChain, self).__init__(name, beads)
        for bead1, bead2 in zip(self._beads[:-1], self._beads[1:]):
            self.add_bond(bead1, bead2)


class Topology(object):

    def __init__(self):
        self._chains = []
        self._n_chains = 0
        self._n_beads = 0
        self._bonds = []
        self._box_lengths = None

    def add_chain(self, chain, n=1):
        for _ in range(n):
            # copy chain
            chain_copy = deepcopy(chain)

            # update indices of each bead
            for bead in chain_copy.beads:
                bead.index = bead.index_in_chain + self._n_beads

            # update index of chain
            chain_copy.index = self._n_chains

            # add chain and bonds to topology
            self._chains.append(chain_copy)
            for bond in chain_copy.bonds:
                self._bonds.append(bond)

            # update chain and bead counts
            self._n_chains += 1
            self._n_beads += chain_copy.n_beads

    @property
    def chains(self):
        return iter(self._chains)

    @property
    def beads(self):
        for chain in self.chains:
            for bead in chain.beads:
                yield bead

    @property
    def bonds(self):
        return iter(self._bonds)

    @property
    def n_chains(self):
        return self._n_chains

    @property
    def n_beads(self):
        return self._n_beads

    @property
    def n_bonds(self):
        return len(self._bonds)

    @property
    def box_lengths(self):
        return self._box_lengths

    @box_lengths.setter
    def box_lengths(self, new_box_lengths):
        self._box_lengths = new_box_lengths
