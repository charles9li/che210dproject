import numpy as np
from numba import jit


class System(object):

    def __init__(self, topology, force_field):
        self.topology = topology
        self.force_field = force_field
        self._force_energy_function = None

    def initialize(self):
        self.force_field.initialize()
        self._create_force_energy_function()

    def _create_force_energy_function(self):
        # get potentials from force field
        wall_ljwca_potential = self.force_field.wall_ljwca_potential
        ljwca_potential = self.force_field.ljwca_potential
        harmonic_bond_potential = self.force_field.harmonic_bond_potential

        # determine if whether to compute a potential
        compute_wall_ljwca = wall_ljwca_potential is not None
        compute_ljwca = ljwca_potential is not None
        compute_harmonic_bond = harmonic_bond_potential is not None

        # get the functions used to compute forces and energies from each potential
        wall_ljwca_force_energy_function = wall_ljwca_potential.force_energy_function if compute_wall_ljwca else None
        ljwca_force_energy_function = ljwca_potential.force_energy_function if compute_ljwca else None
        harmonic_bond_force_energy_function = harmonic_bond_potential.force_energy_function if compute_harmonic_bond else None

        # get the relevant parameters about the system
        bead_names_in_force_field = list(self.force_field.bead_names)
        n_beads = self.topology.n_beads
        box_lengths = self.topology.box_lengths
        periodicity = np.array(self.topology.periodicity, dtype=float)
        bead_types = [b.bead_type.name for b in self.topology.beads]
        bead_species_indices = np.array([bead_names_in_force_field.index(bt) for bt in bead_types])

        # create bonded matrix
        is_bonded = np.zeros((n_beads, n_beads), dtype=bool)
        for bond in self.topology.bonds:
            bead1, bead2 = bond
            is_bonded[bead1.index, bead2.index] = True
            is_bonded[bead2.index, bead1.index] = True

        @jit(nopython=True)
        def calculate_force_energy(positions):
            forces = np.zeros_like(positions)
            potential_energy = 0.0
            for i in range(n_beads):
                r_i = positions[i, :]
                species_index_i = bead_species_indices[i]

                # compute LJWCA wall
                if compute_wall_ljwca:
                    forces, potential_energy = wall_ljwca_force_energy_function(forces, potential_energy,
                                                                                r_i, i, species_index_i)

                r_ij_array = positions[:i, :] - r_i
                r_ij_array = r_ij_array - periodicity * box_lengths * np.rint(r_ij_array / box_lengths)
                d_sqd_array = np.sum(r_ij_array * r_ij_array, axis=1)
                for j in range(i):
                    d_sqd = d_sqd_array[j]
                    r_ij = r_ij_array[j]
                    species_index_j = bead_species_indices[j]

                    # compute harmonic bond
                    if compute_harmonic_bond and is_bonded[i, j]:
                        forces, potential_energy = harmonic_bond_force_energy_function(forces, potential_energy,
                                                                                       r_ij, d_sqd, i, j,
                                                                                       species_index_i, species_index_j)

                    # compute LJWCA
                    if compute_ljwca:
                        forces, potential_energy = ljwca_force_energy_function(forces, potential_energy,
                                                                               r_ij, d_sqd, i, j,
                                                                               species_index_i, species_index_j)
            return forces, potential_energy

        self._force_energy_function = calculate_force_energy
        return calculate_force_energy

    @property
    def force_energy_function(self):
        return self._force_energy_function
