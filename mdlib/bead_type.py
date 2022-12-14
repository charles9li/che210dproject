"""bead_type.py: Used for managing bead types."""
__all__ = ['BeadType']


class BeadType(object):
    """A BeadType represents a """

    _bead_type_by_name = {}
    n_bead_types = 0

    def __init__(self, name, mass=1.0):
        # name of the bead type
        self._name = name
        # mass of the bead type
        self._mass = mass

        # check if bead type already exists
        if name in BeadType._bead_type_by_name:
            raise ValueError(f"Duplicate bead type {name}")

        # update index of this BeadType and number of bead types
        self._index = BeadType.n_bead_types
        BeadType.n_bead_types += 1

        # add to dictionary
        BeadType._bead_type_by_name[name] = self

    @staticmethod
    def get_by_name(name):
        """Get the BeadType with the particular name."""
        n = name.strip()
        return BeadType._bead_type_by_name[n]

    @staticmethod
    def has_bead_type(name):
        """Checks if BeadType with specified name exists."""
        n = name.strip()
        return n in BeadType._bead_type_by_name

    @property
    def name(self):
        return self._name

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, new_mass):
        self._mass = new_mass

    @property
    def index(self):
        return self._index
