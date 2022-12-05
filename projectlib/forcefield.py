"""forcefield.py: Contains the ForceField class which stores """
from projectlib.bead_type import BeadType
from projectlib.potentials import HarmonicBias, WallLJWCA, LJWCA, HarmonicBond


class ForceField(object):
    """Stores information about the BeadTypes and potentials in a system.

    Attributes
    ----------
    system : projectlib.System
        Stores information about the molecular constituents in the system. This
        attribute is set when this ForceField is added to a System.
    """

    def __init__(self):
        self.system = None

        self.harmonic_bias = None
        self.wall_ljwca_potential = None
        self.ljwca_potential = None
        self.harmonic_bond_potential = None

    def add_potential(self, potential):
        """Adds a potential to the ForceField.

        Parameters
        ----------
        potential : projectlib.potentials._Potential
        """
        if isinstance(potential, HarmonicBias):
            self.harmonic_bias = potential
        if isinstance(potential, WallLJWCA):
            self.wall_ljwca_potential = potential
        elif isinstance(potential, LJWCA):
            self.ljwca_potential = potential
        elif isinstance(potential, HarmonicBond):
            self.harmonic_bond_potential = potential
        potential.force_field = self

    def initialize(self):
        for p in self.potentials:
            p.initialize()

    @property
    def potentials(self):
        _potentials = []
        for p in [self.harmonic_bias, self.wall_ljwca_potential, self.ljwca_potential, self.harmonic_bond_potential]:
            if p is not None:
                _potentials.append(p)
        return _potentials

    @property
    def bead_types(self):
        """Iterator over BeadTypes in the ForceField."""
        # get bead types from all interactions
        _bead_types = []
        for potential in self.potentials:
            if not isinstance(potential, HarmonicBias):
                for interaction in potential.interactions:
                    for bead_name in interaction:
                        _bead_types.append(BeadType.get_by_name(bead_name))

        # get unique bead types
        _bead_types = list(set(_bead_types))

        # order by index of bead types
        _bead_types = sorted(_bead_types, key=lambda bead_type: bead_type.index)

        return iter(_bead_types)

    @property
    def bead_names(self):
        """Iterator over names of BeadTypes in the ForceField."""
        for bead_type in self.bead_types:
            yield bead_type.name

    @property
    def n_bead_types(self):
        """Number of BeadTypes in the ForceField."""
        return len(list(self.bead_types))
