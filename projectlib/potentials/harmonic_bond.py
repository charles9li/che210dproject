import numpy as np
from numba import jit

from ._potential_base_classes import _BondedPairPotential


class HarmonicBond(_BondedPairPotential):
    PARAMETERS = ['k', 'r0']
    ALL_PARAMETERS = PARAMETERS

    def __init__(self):
        super(HarmonicBond, self).__init__()

    def add_interaction(
            self, bead_name_1, bead_name_2,
            k=3000.0, r0=1.0,
    ):
        super(HarmonicBond, self).add_interaction(
            bead_name_1, bead_name_2,
            k=k, r0=r0
        )

    def create_force_energy_function(self):
        self._create_if_compute_and_parameter_arrays()
        if_compute = self._if_compute
        parameter_array = self._parameter_array

        _k_index = self._get_parameter_index('k')
        _r0_index = self._get_parameter_index('r0')

        @jit(nopython=True)
        def calculate_force_energy(
                forces, potential_energy,
                r_ij, d_sqd,
                i, j,
                species_index_i, species_index_j
        ):
            if not if_compute[species_index_i, species_index_j]:
                return forces, potential_energy

            # get values of parameters
            k = parameter_array[species_index_i, species_index_j, _k_index]
            r0 = parameter_array[species_index_i, species_index_j, _r0_index]

            # calculate distance
            d = np.sqrt(d_sqd)

            # calculate force and potential energy
            force_ij = r_ij * k * (1. - r0 / max(d, 1.e-300))
            forces[i, :] += force_ij
            forces[j, :] -= force_ij
            dr = d - r0
            potential_energy += 0.5 * k * dr * dr

            return forces, potential_energy

        self._force_energy_function = calculate_force_energy
        return calculate_force_energy
