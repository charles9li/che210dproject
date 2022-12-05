import numpy as np
from numba import jit

from ._potential_base_classes import _Potential


class HarmonicBias(_Potential):
    PARAMETER_NAMES = ['k', 'r0', 'axis']
    PARAMETER_TYPES = [float, float, int]

    def __init__(self):
        super(HarmonicBias).__init__()
        self._parameters = []

    def add_interaction(self, group, k=10.0, r0=0.0, axis=0):
        self._parameters.append((np.array(group, dtype=int), {'k': k, 'r0': r0, 'axis': 0}))
        return len(self._parameters) - 1

    def set_parameter_value(self, parameter_name, new_value, group_index, *args, **kwargs):
        # check that parameter exists in this potential
        self._raise_no_parameter_error(parameter_name)

        # update value
        self._parameters[group_index][1][parameter_name] = new_value

    def get_parameter_value(self, parameter_name, group_index):
        # check that parameter exists in this potential
        self._raise_no_parameter_error(parameter_name)

        return self._parameters[group_index][1][parameter_name]

    def initialize(self):
        self._create_force_energy_function()

    def _create_force_energy_function(self):
        _n_groups = self.n_groups
        _groups = [p[0] for p in self._parameters]
        _each_group_len = np.array([len(g) for g in _groups], dtype=int)
        _max_len = max(_each_group_len)

        _each_group_indices = np.zeros((self.n_groups, _max_len), dtype=int)
        for g, len_g in zip(_groups, _each_group_len):
            _each_group_indices[:len_g] = g

        _k_array = np.array([self.get_parameter_value('k', i) for i in range(self.n_groups)], dtype=float)
        _r0_array = np.array([self.get_parameter_value('r0', i) for i in range(self.n_groups)], dtype=float)
        _axis_array = np.array([self.get_parameter_value('axis', i) for i in range(self.n_groups)], dtype=int)

        @jit(nopython=True)
        def calculate_force_energy(forces, potential_energy, positions):
            for _g_index in range(_n_groups):
                _g_len = _each_group_len[_g_index]
                _bead_indices = _each_group_indices[_g_index, :_g_len]
                _group_positions = np.empty((_g_len, 3), dtype=float)
                for i, _b_index in enumerate(_bead_indices):
                    _group_positions[i, :] = positions[_b_index, :]
                _center_position = np.mean(_group_positions, axis=1)
                k = _k_array[_g_index]
                r0 = _r0_array[_g_index]
                axis = _axis_array[_g_index]
                dr = (_center_position[axis] - r0)
                potential_energy += 0.5 * k * dr**2
                force = -k * dr / float(_g_len)
                for _b_index in _bead_indices:
                    force[_b_index, axis] += force
            return forces, potential_energy

        self._force_energy_function = calculate_force_energy

    @property
    def n_groups(self):
        return len(self._parameters)
