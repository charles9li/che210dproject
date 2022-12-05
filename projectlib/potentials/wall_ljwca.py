import numpy as np
from numba import jit

from ._potential_base_classes import _Potential


class WallLJWCA(_Potential):
    PARAMETER_NAMES = ['eps', 'sigma', 'cut', 'axis', 'lower_bound', 'upper_bound', 'lambda_lj', 'lambda_wca']
    PARAMETER_TYPES = [float, float, float, int, float, float, float, float]
    DEPENDENT_PARAMETER_NAMES = ['shift_lj']
    DEPENDENT_PARAMETER_TYPES = [float]

    def add_interaction(
            self, bead_name, eps=1.0, sigma=1.0, cut=3.0,
            axis=0, lower_bound=0.0, upper_bound=1.0,
            lambda_lj=1.0, lambda_wca=0.0
    ):
        super(WallLJWCA, self).add_interaction(
            bead_name, eps=eps, sigma=sigma, cut=cut,
            axis=axis, lower_bound=lower_bound, upper_bound=upper_bound,
            lambda_lj=lambda_lj, lambda_wca=lambda_wca
        )

    def _update_dependent_parameters(self, bead_name):
        # get current values of the relevant parameters
        eps = self.get_parameter_value('eps', bead_name)
        sigma = self.get_parameter_value('sigma', bead_name)
        cut = self.get_parameter_value('cut', bead_name)

        # compute the shift factor
        shift_lj = -1.5 * np.sqrt(3) * eps * ((sigma / cut)**9 - (sigma / cut)**3)

        # update values of the parameters
        dependent_parameter_values = {'shift_lj': shift_lj}
        for parameter_name, value in dependent_parameter_values.items():
            self.set_parameter_value(parameter_name, value, bead_name, update_dependents=False)

    def _create_force_energy_function(self):
        _if_compute = self._interaction_array

        _eps_array = self._parameter_arrays['eps']
        _sigma_array = self._parameter_arrays['sigma']
        _cut_array = self._parameter_arrays['cut']
        _axis_array = self._parameter_arrays['axis']
        _lower_bound_array = self._parameter_arrays['lower_bound']
        _upper_bound_array = self._parameter_arrays['upper_bound']
        _shift_lj_array = self._parameter_arrays['shift_lj']
        _lambda_lj_array = self._parameter_arrays['lambda_lj']
        _lambda_wca_array = self._parameter_arrays['lambda_wca']

        energy_coefficient = 1.5 * np.sqrt(3.)
        force_coefficient = 3. * energy_coefficient
        wca_cut_coefficient = 3.**(1./6.)

        @jit(nopython=True)
        def calculate_force_energy(
                forces, potential_energy,
                r_i, i, species_index
        ):
            # return if not computing
            if not _if_compute[species_index]:
                return forces, potential_energy

            # get parameters for this interaction
            eps = _eps_array[species_index]
            sigma = _sigma_array[species_index]
            cut = _cut_array[species_index]
            axis = _axis_array[species_index]
            lower_bound = _lower_bound_array[species_index]
            upper_bound = _upper_bound_array[species_index]
            shift_lj = _shift_lj_array[species_index]
            lambda_lj = _lambda_lj_array[species_index]
            lambda_wca = _lambda_wca_array[species_index]

            # do each bound
            for force_sign, bound in zip([1.0, -1.0], [lower_bound, upper_bound]):
                # distance from the surface
                if force_sign == 1.0:
                    d = r_i[axis] - bound
                else:
                    d = bound - r_i[axis]

                # don't compute if outside LJ cut
                if d > cut:
                    continue

                # determine whether to compute LJ and/or WCA
                compute_lj = lambda_lj != 0.0
                compute_wca = (lambda_wca != 0.0) and d < wca_cut_coefficient * sigma

                # continue if not computing either
                if not compute_lj and not compute_wca:
                    continue

                # compute inverse distances
                inv_d = sigma / d
                inv_d3 = inv_d * inv_d * inv_d
                inv_d4 = inv_d * inv_d3
                inv_d9 = inv_d3 * inv_d3 * inv_d3
                inv_d10 = inv_d * inv_d9

                # compute LJ forces and potential
                f_i = force_coefficient * eps / sigma * (3. * inv_d10 - inv_d4)
                u_i = energy_coefficient * eps * (inv_d9 - inv_d3)

                # update with cut and shifted LJ
                if compute_lj:
                    forces[i, axis] += force_sign * lambda_lj * f_i
                    potential_energy += lambda_lj * (u_i + shift_lj)

                # update with value of repulsive WCA
                if compute_wca:
                    forces[i, axis] += force_sign * lambda_wca * f_i
                    potential_energy += lambda_wca * (u_i + eps)

            return forces, potential_energy

        self._force_energy_function = calculate_force_energy
        return calculate_force_energy
