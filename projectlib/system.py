import numpy as np
from numba import jit


class System(object):

    def __init__(self, box_lengths, cutoff=2.5):
        self._force_field = None
        self.box_lengths = box_lengths
        self.cutoff = cutoff
        self._bead_types = {}
        self._chains = {}
        self._interactions = {}

        self._num_beads = None
        self._bead_species = None
        self._bead_species_indices = None
        self._chain_names = None
        self._chain_indices = None
        self._force_energy_function = None

    @property
    def force_field(self):
        return self._force_field

    @force_field.setter
    def force_field(self, force_field):
        self._force_field = force_field
        force_field.system = self

    def add_bead_type(self, bead_type):
        self._bead_types[bead_type.name] = bead_type

    def get_bead_type(self, bead_name):
        return self._bead_types[bead_name]

    def add_chain(self, chain, num=1):
        self._chains[chain.name] = [chain, num]

    def get_chain(self, chain_name):
        return self._chains[chain_name][0]

    def get_chain_num(self, chain_name):
        return self._chains[chain_name][1]

    def set_chain_num(self, chain_name, num=1):
        self._chains[chain_name][1] = num

    @property
    def n_beads(self):
        return sum([chain.n_beads * num for chain, num in self._chains.values()])

    @property
    def all_bead_names(self):
        all_bead_names = []
        for chain, num in self._chains.values():
            for _ in range(num):
                all_bead_names.extend(chain.bead_names)
        return np.array(all_bead_names, dtype=str)

    @property
    def all_chain_names(self):
        all_chain_names = []
        for chain, num in self._chains.values():
            for _ in range(num):
                all_chain_names.append(chain.name)
        return np.array(all_chain_names, dtype=str)

    # def create_force_energy_function(self):
    #     # loop through molecules and create lists of bead names and bonds
    #     bead_names = []
    #     chain_names = []
    #     chain_indices = []
    #     all_bonds = []
    #     current_chain_index = 0
    #     for chain, num in self._chains.values():
    #         chain_names.append(chain.name)
    #         for _ in range(num):
    #             num_beads_curr = len(bead_names)
    #             bead_names.extend(chain.bead_names)
    #             chain_indices.append(current_chain_index)
    #             all_bonds.extend(map(lambda x: (x[0]+num_beads_curr, x[1]+num_beads_curr), chain.bonds))
    #             current_chain_index += 1
    #     bead_names = np.array(bead_names, dtype=str)
    #     forcefield_bead_names = list(self.force_field.bead_names)
    #     bead_species_indices = np.array([forcefield_bead_names.index(bn) for bn in bead_names], dtype=int)
    #
    #     # create bonded matrix
    #     bonded_matrix = np.full((self.n_beads, self.n_beads), False)
    #     for bond in all_bonds:
    #         i, j = bond
    #         bonded_matrix[i, j] = bonded_matrix[j, i] = True
    #
    #     cutoff = self.cutoff
    #     box_lengths = self.box_lengths
    #     num_beads = self.n_beads
    #
    #     default_force = np.zeros(3)
    #
    #     @jit(nopython=True)
    #     def ljts(forces, potential_energy, i, j, r_ij, d_sqd, shift, cutoff_sqd):
    #         # if d_sqd > cutoff_sqd:
    #         #     return forces, potential_energy
    #         inv_d2 = 1. / d_sqd
    #         inv_d6 = inv_d2 * inv_d2 * inv_d2
    #         inv_d12 = inv_d6 * inv_d6
    #         force_ij = r_ij * ((-48. * inv_d12 + 24. * inv_d6) * inv_d2)
    #         u_ij = 4. * (inv_d12 - inv_d6) + shift
    #         forces[i, :] += force_ij
    #         forces[j, :] -= force_ij
    #         potential_energy += u_ij
    #         return forces, potential_energy
    #
    #     @jit(nopython=True)
    #     def force_energy_function(positions):
    #         cutoff_sqd = cutoff * cutoff
    #         shift = -4. * (cutoff**-12 - cutoff**-6)
    #         forces = np.zeros_like(positions)
    #         potential_energy = 0.0
    #         for i in range(num_beads):
    #             r_ij_array = positions[:i, :] - positions[i, :]
    #             r_ij_array = r_ij_array - box_lengths * np.rint(r_ij_array / box_lengths)
    #             d_sqd_array = np.sum(r_ij_array * r_ij_array, axis=1)
    #             for j in range(i):
    #                 d_sqd = d_sqd_array[j]
    #                 r_ij = r_ij_array[j]
    #
    #                 if d_sqd > cutoff_sqd:
    #                     continue
    #                 # inv_d2 = 1. / d_sqd
    #                 # inv_d6 = inv_d2 * inv_d2 * inv_d2
    #                 # inv_d12 = inv_d6 * inv_d6
    #                 # force_ij = r_ij * ((-48. * inv_d12 + 24. * inv_d6) * inv_d2)
    #                 # forces[i, :] += force_ij
    #                 # forces[j, :] -= force_ij
    #                 # potential_energy += 4. * (inv_d12 - inv_d6) + shift
    #
    #                 forces, potential_energy = ljts(forces, potential_energy, i, j, r_ij, d_sqd, shift, cutoff_sqd)
    #
    #         return forces, potential_energy
    #
    #     self._force_energy_function = force_energy_function
    #     return force_energy_function

    def create_force_energy_function(self):
        # loop through molecules and create lists of bead names and bonds
        bead_names = []
        chain_names = []
        chain_indices = []
        all_bonds = []
        current_chain_index = 0
        for chain, num in self._chains.values():
            chain_names.append(chain.name)
            for _ in range(num):
                num_beads_curr = len(bead_names)
                bead_names.extend(chain.bead_names)
                chain_indices.append(current_chain_index)
                all_bonds.extend(map(lambda x: (x[0]+num_beads_curr, x[1]+num_beads_curr), chain.bonds))
                current_chain_index += 1
        bead_names = np.array(bead_names, dtype=str)
        forcefield_bead_names = list(self.force_field.bead_names)
        bead_species_indices = np.array([forcefield_bead_names.index(bn) for bn in bead_names], dtype=int)

        # create bonded matrix
        bonded_matrix = np.full((self.n_beads, self.n_beads), False)
        for bond in all_bonds:
            i, j = bond
            bonded_matrix[i, j] = bonded_matrix[j, i] = True

        lj_force_energy = self.force_field._potentials[2][0].force_energy_function

        box_lengths = self.box_lengths
        num_beads = self.n_beads

        @jit(nopython=True)
        def force_energy_function(positions):
            forces = np.zeros_like(positions)
            potential_energy = 0.0
            for i in range(num_beads):
                position_i = positions[i, :]
                r_ij_array = positions[:i, :] - position_i
                r_ij_array = r_ij_array - box_lengths * np.rint(r_ij_array / box_lengths)
                d_sqd_array = np.sum(r_ij_array * r_ij_array, axis=1)
                for j in range(i):
                    d_sqd = d_sqd_array[j]
                    r_ij = r_ij_array[j]
                    forces, potential_energy = lj_force_energy(forces, potential_energy, r_ij, d_sqd, i, j, bead_species_indices[i], bead_species_indices[j])

            return forces, potential_energy

        self._force_energy_function = force_energy_function
        return force_energy_function

    @property
    def force_energy_function(self):
        if self._force_energy_function is None:
            self.create_force_energy_function()
        return self._force_energy_function
