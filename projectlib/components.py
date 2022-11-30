"""components.py: Contains classes that make up the topology of a system."""
__all__ = ['BeadType', 'Chain', 'LinearChain']


class BeadType(object):

    def __init__(self, name, mass=1.0, sigma=1.0):
        self.name = name
        self.mass = mass
        self.sigma = sigma


class Chain(object):

    def __init__(self, name, bead_names):
        self.name = name
        self.bead_names = bead_names
        self.bonds = []

    @property
    def n_beads(self):
        return len(self.bead_names)

    def add_bond(self, bead_index_1, bead_index_2):
        self.bonds.append((bead_index_1, bead_index_2))


class LinearChain(Chain):

    def __init__(self, name, bead_names):
        super(LinearChain, self).__init__(name, bead_names)
        for i in range(len(self.bead_names) - 1):
            self.add_bond(i, i+1)
