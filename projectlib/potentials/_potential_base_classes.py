"""_potential_base_classes.py: Contains various base classes that all potentials inherit from."""
__all__ = ['_Potential', '_PairPotential', '_BondedPairPotential']

from collections import defaultdict
import itertools

import numpy as np
from numba import jit

from projectlib.bead_type import BeadType


# =========================================================================
# Utility functions used for working with nested dictionaries
# =========================================================================

def _create_empty_nested_dict(n_nests):
    if n_nests == 1:
        return dict()
    return defaultdict(lambda: _create_empty_nested_dict(n_nests - 1))


def _get_value_from_nested_dict(nested_dict, *keys):
    first_key = keys[0]
    if len(keys) == 1:
        return nested_dict[first_key]
    return _get_value_from_nested_dict(nested_dict[first_key], *keys[1:])


def _set_value_in_parameter_dict(nested_dict, new_value, *keys):
    first_key = keys[0]
    if len(keys) == 1:
        nested_dict[first_key] = new_value
        return
    _set_value_in_parameter_dict(nested_dict[first_key], new_value, *keys[1:])


def _get_key_groups_from_nested_dict(nested_dict, depth):
    key_groups = []
    for key in nested_dict.keys():
        if depth == 1:
            key_groups.append([key])
        else:
            for kg in _get_key_groups_from_nested_dict(nested_dict, depth-1):
                key_groups.append([key] + list(kg))
    return list(map(lambda x: tuple(x), key_groups))


# =========================================================================
# _Potential base class definition
# =========================================================================

class _Potential(object):
    """Base class for all potentials.

    Attributes
    ----------
    N_BEADS : int
        Class attribute that specifies the number of beads involved in each
        interaction.
    BONDED : bool
        Specifies whether this potential is bonded. If True, then the order of
        the bead types matters when specifying interactions.
    PARAMETER_NAMES : list of str
        List of names of independent parameters for each interaction.
    PARAMETER_TYPES : list of numba.core.types.abstract._TypeMetaclass
        List of Numba types of the independent parameters.
    DEPENDENT_PARAMETER_NAMES : list of str
        List of names of parameters that depend on the independent parameters.
    DEPENDENT_PARAMETER_TYPES : list of list of numba.core.types.abstract._TypeMetaclass
        List of Numba types of the dependent parameters.
    """
    N_BEADS = 1
    BONDED = False
    PARAMETER_NAMES = []
    PARAMETER_TYPES = []
    DEPENDENT_PARAMETER_NAMES = []
    DEPENDENT_PARAMETER_TYPES = []
    
    def __new__(cls, *args, **kwargs):
        cls.ALL_PARAMETER_NAMES = cls.PARAMETER_NAMES + cls.DEPENDENT_PARAMETER_NAMES
        cls.ALL_PARAMETER_TYPES = cls.PARAMETER_TYPES + cls.DEPENDENT_PARAMETER_TYPES
        cls.N_PARAMETERS = len(cls.PARAMETER_NAMES + cls.DEPENDENT_PARAMETER_NAMES)
        return super(_Potential, cls).__new__(cls, *args, **kwargs)

    def __init__(self):
        # initialize list of interactions and dictionary that holds all the parameters
        self._interactions = []
        self._parameters = dict()
        for parameter_name, parameter_type in zip(self.ALL_PARAMETER_NAMES, self.ALL_PARAMETER_TYPES):
            self._parameters[parameter_name] = _create_empty_nested_dict(self.N_BEADS)

        # initialize force field attribute that will be set when this potential
        # is added to a force field
        self.force_field = None

        # initialize attributes that are only set after compilation
        self._interaction_array = None
        self._parameter_arrays = None
        self._force_energy_function = None

    def add_interaction(self, *bead_types, **kwargs):
        """Adds an interaction between specified bead types to this potential.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.
        **kwargs : dict of str: float
            Values of parameters in the interaction.
        """
        # check that provided number of bead names matches the potential
        self._check_n_bead_names(*bead_types)

        # check that bead types exist and convert to str
        bead_types = self._validate_and_convert_bead_types(*bead_types)

        # add interactions to the private list
        # if bonded, order of bead names matters
        if self.BONDED:
            self._interactions.append(tuple(bead_types))
            self._interactions.append(tuple(reversed(bead_types)))
        # if non bonded, order doesn't matter
        else:
            for p in set(itertools.permutations(bead_types)):
                self._interactions.append(tuple(p))

        # set values of all parameters specified in kwargs
        for parameter_name, value in kwargs.items():
            self.set_parameter_value(parameter_name, value, *bead_types, update_dependents=False)

        # update values of dependent parameters
        self._update_dependent_parameters(*bead_types)

    def has_interaction(self, *bead_types):
        """Checks if this potential has an interaction between the specified
        bead names.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.
        """
        # check that provided number of bead names matches the potential
        self._check_n_bead_names(*bead_types)

        # check that bead types exist and convert to str
        bead_types = self._validate_and_convert_bead_types(*bead_types)

        # check that bead_types exist in private attribute
        return tuple(bead_types) in self._interactions

    def has_parameter(self, parameter_name):
        """Returns true if this potential has a parameter of the specified
        name.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Returns
        -------
        bool
            True if this potential has the specified parameter.
        """
        return parameter_name in self.ALL_PARAMETER_NAMES

    def set_parameter_value(
            self, parameter_name, new_value, *bead_types,
            update_dependents=True
    ):
        """Set value of a parameter for an interaction between the specified
        bead types.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.
        new_value : float
            New value of the parameter.
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.
        update_dependents : bool, default=True
            Flag that specifies whether dependent parameters should be updated.
            Should always be True unless you know what you're doing.
        """
        # check that provided number of bead names matches the potential
        self._check_n_bead_names(*bead_types)

        # check that bead types exist and convert to str
        bead_types = self._validate_and_convert_bead_types(*bead_types)

        # check that parameter exists in this potential
        self._raise_no_parameter_error(parameter_name)

        # raise error if no interaction exists for specified bead names
        self._raise_no_interaction_error(*bead_types)

        # can't set values of dependent parameters unless forcibly overriden
        if update_dependents and parameter_name in self.DEPENDENT_PARAMETER_NAMES:
            raise ValueError("can't set values of dependent parameters")

        # set new value of the parameter
        _parameter_dict = self._parameters[parameter_name]
        _parameter_type = self.get_parameter_type(parameter_name)
        # if bonded, order of bead names matters
        if self.BONDED:
            _set_value_in_parameter_dict(_parameter_dict, _parameter_type(new_value), *bead_types)
            _set_value_in_parameter_dict(_parameter_dict, _parameter_type(new_value), *reversed(bead_types))
        # if non bonded, order doesn't matter
        else:
            for p in set(itertools.permutations(bead_types)):
                _set_value_in_parameter_dict(_parameter_dict, _parameter_type(new_value), *p)

        # update dependent parameters unless forcibly overriden
        if update_dependents:
            self._update_dependent_parameters(*bead_types)

    def get_parameter_value(self, parameter_name, *bead_types):
        """Get value of a parameter for an interaction between the specified
        bead types.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.

        Returns
        -------
        float
            Value of the parameter.
        """
        # check that provided number of bead names matches the potential
        self._check_n_bead_names(*bead_types)

        # check that bead types exist and convert to str
        bead_types = self._validate_and_convert_bead_types(*bead_types)

        # check that parameter exists in this potential
        self._raise_no_parameter_error(parameter_name)

        # raise error if no interaction exists for specified bead names
        self._raise_no_interaction_error(*bead_types)

        _parameter_dict = self._parameters[parameter_name]
        return _get_value_from_nested_dict(_parameter_dict, *bead_types)

    def _get_parameter_index(self, parameter_name):
        """Private helper method that returns the index for a specified
        parameter.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Returns
        -------
        int
            Index of the parameter.
        """
        return self.ALL_PARAMETER_NAMES.index(parameter_name)

    def get_parameter_type(self, parameter_name):
        """Returns the type of specified parameter.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Returns
        -------
        parameter_type
        """
        _parameter_index = self._get_parameter_index(parameter_name)
        return self.ALL_PARAMETER_TYPES[_parameter_index]

    @staticmethod
    def _validate_and_convert_bead_types(*bead_types):
        """Private helper method that converts list of BeadTypes to a tuple of
        str.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.

        Returns
        -------
        bead_names : tuple of str
            Tuple of names of BeadTypes.
        """
        bead_names = []
        for bt in bead_types:
            if isinstance(bt, str):
                if BeadType.has_bead_type(bt):
                    bead_names.append(bt)
                else:
                    raise ValueError(f"BeadType with name {bt} doesn't exist")
            else:
                bead_names.append(bt.name)
        return tuple(bead_names)

    def _update_dependent_parameters(self, *bead_types):
        """Private helper method that updates the values of all dependent
        parameters for a specified interaction.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.
        """
        pass

    def _check_n_bead_names(self, *bead_types):
        """Raises an error if number of provided bead names doesn't match the
        number of bead types in each interaction for this potential.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.

        Raises
        ------
        ValueError
            If there is a mismatch between the number of bead names specified
            and the number of bead types involved in each interaction.
        """
        if len(bead_types) != self.N_BEADS:
            raise ValueError(
                f"number of bead names provided does not match number of beads "
                f"involved in this interaction, which is {self.N_BEADS}"
            )

    def _raise_no_interaction_error(self, *bead_types):
        """Raises an error if no interaction exists between the specified bead
        names in this potential.

        Parameters
        ----------
        *bead_types : str or projectlib.BeadType
            Variable length tuple of bead names or BeadTypes.

        Raises
        ------
        ValueError
            When no interaction exists between the specified bead names.
        """
        if not self.has_interaction(*bead_types):
            raise ValueError(f"no interaction exists between species ({', '.join(bead_types)}) in this potential")

    def _raise_no_parameter_error(self, parameter_name):
        """Raises an error if no parameter with the specified name exists in
        this potential.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Raises
        ------
        ValueError
            When no interaction exists between the specified bead names.
        """
        if not self.has_parameter(parameter_name):
            raise ValueError(f"no parameter '{parameter_name}' exists in this potential")

    def initialize(self):
        """Initializes the interaction by creating the function that computes
        forces and energies.
        """
        # construct parameter and interaction arrays
        self._convert_parameters_to_arrays()
        self._convert_interactions_to_array()

        # create force energy function
        self._create_force_energy_function()

    def _create_force_energy_function(self):
        """Factory method that creates a function to calculate the force and
        energy due to this potential. Child classes will override this method.
        """
        @jit(nopython=True)
        def calculate_force_energy(forces, potential_energy):
            return forces, potential_energy
        self._force_energy_function = calculate_force_energy

    @property
    def force_energy_function(self):
        """Numba-accelerated function that computes the force and energy due
        to this potential."""
        return self._force_energy_function

    @property
    def interactions(self):
        """All interactions in this potential."""
        return self._interactions

    def _convert_parameters_to_arrays(self):
        """Private helper method that converts a nested dict for a parameter
        into an array where the indices correspond to the index of the bead
        type in the force field.
        """
        # get bead names from force field
        bead_names_in_force_field = list(self.force_field.bead_names)
        n_bead_types_in_force_field = self.force_field.n_bead_types

        # initialize _parameter_arrays
        self._parameter_arrays = dict()

        # create entry for each parameter
        for _parameter_name, _parameter_dict in self._parameters.items():
            # initialize parameter array
            shape = tuple([n_bead_types_in_force_field] * self.N_BEADS)
            _parameter_type = self.get_parameter_type(_parameter_name)
            _parameter_array = np.zeros(shape, dtype=_parameter_type)

            # fill in values of the parameter array
            _parameter_dict = self._parameters[_parameter_name]
            for bead_names in _get_key_groups_from_nested_dict(_parameter_dict, self.N_BEADS):
                value = _get_value_from_nested_dict(_parameter_dict, *bead_names)
                indices = tuple([bead_names_in_force_field.index(bn) for bn in bead_names])
                _parameter_array[indices] = value

            self._parameter_arrays[_parameter_name] = _parameter_array

    def _convert_interactions_to_array(self):
        """Private helper method that converts the private interactions
        into an array that specifies whether each interaction exists.
        """
        # get bead names from force field
        bead_names_in_force_field = list(self.force_field.bead_names)
        n_bead_types_in_force_field = self.force_field.n_bead_types

        # initialize array
        shape = tuple([n_bead_types_in_force_field] * self.N_BEADS)
        self._interaction_array = np.zeros(shape, dtype=bool)

        # fill in array
        for _interaction in self._interactions:
            indices = tuple([bead_names_in_force_field.index(bn) for bn in _interaction])
            self._interaction_array[indices] = True


# =========================================================================
# Other base classes for convenience
# =========================================================================

class _PairPotential(_Potential):
    N_BEADS = 2


class _BondedPairPotential(_PairPotential):
    BONDED = True


if __name__ == '__main__':
    pass
