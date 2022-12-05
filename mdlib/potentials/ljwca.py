"""ljwca.py: Contains the LJWCA class that implements a linear combination of
the truncated and shift Lennard-Jones (LJTS) and repulsive
Weeks-Chandler-Andersen (WCA) potentials.
"""
from numba import jit

from ._potential_base_classes import _PairPotential


class LJWCA(_PairPotential):
    """Linear combination of the LJTS and repulsive WCA potentials."""
    PARAMETER_NAMES = ['eps', 'sigma', 'cut', 'lambda_lj', 'lambda_wca']
    PARAMETER_TYPES = [float] * len(PARAMETER_NAMES)
    DEPENDENT_PARAMETER_NAMES = ['sigma2', 'cut2', 'shift_lj']
    DEPENDENT_PARAMETER_TYPES = [float] * len(DEPENDENT_PARAMETER_NAMES)

    def add_interaction(
            self, bead_name_1, bead_name_2,
            eps=1.0, sigma=1.0, cut=2.5,
            lambda_lj=1.0, lambda_wca=0.0
    ):
        """Adds an interaction between two bead types.

        Parameters
        ----------
        bead_name_1, bead_name_2 : str
            Names of the bead types.
        eps : float, default=1.0
            Depth of the LJ potential well.
        sigma : float, default=1.0
            Distance at which the potential is 0.
        cut : float, default=2.5
            Cutoff distance for the LJ potential. The potential is shifted such
            that u(cut) = 0.
        lambda_lj : float, default=1.0
            Scaling factor for the LJ potential.
        lambda_wca : float, default=0.0
            Scaling factor for the WCA potential.
        """
        super(LJWCA, self).add_interaction(
            bead_name_1, bead_name_2,
            eps=eps, sigma=sigma, cut=cut,
            lambda_lj=lambda_lj, lambda_wca=lambda_wca
        )
        self._update_dependent_parameters(bead_name_1, bead_name_2)

    def _update_dependent_parameters(self, bead_name_1, bead_name_2):
        """Private helper method that updates the LJ shift parameter every time
        parameters are changed.

        Parameters
        ----------
        bead_name_1, bead_name_2 : str
            Names of the bead types involved in the pair interaction.
        """
        # get current values of the relevant parameters
        eps = self.get_parameter_value('eps', bead_name_1, bead_name_2)
        sigma = self.get_parameter_value('sigma', bead_name_1, bead_name_2)
        cut = self.get_parameter_value('cut', bead_name_1, bead_name_2)

        # compute the shift parameter
        shift_lj = -4. * eps * ((sigma / cut)**12 - (sigma / cut)**6)

        # update values of the parameters
        dependent_parameter_values = {'sigma2': sigma**2, 'cut2': cut**2, 'shift_lj': shift_lj}
        for parameter_name, value in dependent_parameter_values.items():
            self.set_parameter_value(parameter_name, value, bead_name_1, bead_name_2, update_dependents=False)

    def _create_force_energy_function(self):
        _if_compute = self._interaction_array

        _eps_array = self._parameter_arrays['eps']
        _sigma2_array = self._parameter_arrays['sigma2']
        _cut2_array = self._parameter_arrays['cut2']
        _shift_lj_array = self._parameter_arrays['shift_lj']
        _lambda_lj_array = self._parameter_arrays['lambda_lj']
        _lambda_wca_array = self._parameter_arrays['lambda_wca']

        @jit(nopython=True)
        def calculate_force_energy(
                forces, potential_energy,
                r_ij, d_sqd,
                i, j,
                species_index_i, species_index_j
        ):
            # return if not computing
            if not _if_compute[species_index_i, species_index_j]:
                return forces, potential_energy

            # get parameters for this interaction
            eps = _eps_array[species_index_i][species_index_j]
            sigma2 = _sigma2_array[species_index_i][species_index_j]
            cut2 = _cut2_array[species_index_i][species_index_j]
            shift_lj = _shift_lj_array[species_index_i][species_index_j]
            lambda_lj = _lambda_lj_array[species_index_i][species_index_j]
            lambda_wca = _lambda_wca_array[species_index_i][species_index_j]

            # return if r > cutoff
            if d_sqd > cut2:
                return forces, potential_energy

            # compute reduced inverse squared and sixth distances
            inv_d2 = sigma2 / d_sqd
            inv_d6 = inv_d2 * inv_d2 * inv_d2

            # determine whether to compute LJ and/or WCA
            compute_lj = lambda_lj != 0.0
            compute_wca = (lambda_wca != 0.0) and inv_d6 > 0.5

            # return if not computing either
            if not compute_lj and not compute_wca:
                return forces, potential_energy

            # compute reduced inverse twelfth distance and LJ forces and potential
            inv_d12 = inv_d6 * inv_d6
            lj_force = r_ij * eps * ((-48. * inv_d12 + 24. * inv_d6) * inv_d2)
            lj_potential = 4. * eps * (inv_d12 - inv_d6)

            # update with value of cut and shifted LJ
            if compute_lj:
                force_ij = lambda_lj * lj_force
                forces[i, :] += force_ij
                forces[j, :] -= force_ij
                potential_energy += lambda_lj * (lj_potential + shift_lj)

            # update with value of repulsive WCA
            if compute_wca:
                force_ij = lambda_wca * lj_force
                forces[i, :] += force_ij
                forces[j, :] -= force_ij
                potential_energy += lambda_wca * (lj_potential + eps)

            return forces, potential_energy

        self._force_energy_function = calculate_force_energy
