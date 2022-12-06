"""simulation.py: Contains classes for running simulations and storing their
states.
"""
__all__ = ['State', 'Simulation']

import os
import time
import random
import string

import numpy as np

import mdtraj as md

from mdlib.minimize_energy import conjugate_gradient
from mdlib import utils


def _generate_temporary_prefix():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))


def initialize_positions(topology, edge_buffer=0.5):
    # create box string
    box = f"{edge_buffer*10.0} {edge_buffer*10.0} {edge_buffer*10.0} " \
          f"{(topology.box_lengths[0]-edge_buffer)*10.0} " \
          f"{(topology.box_lengths[1]-edge_buffer)*10.0} " \
          f"{(topology.box_lengths[2]-edge_buffer)*10.0}"

    # initialize packmol str
    packmol_str = "tolerance 2.0\n"
    output_pdb = _generate_temporary_prefix() + ".pdb"
    packmol_str += f"output {output_pdb}\n"
    packmol_str += f"filetype pdb\n"

    # save temporary chain pdbs for removal later
    chain_pdbs = []
    for chain_type, chain_num in zip(topology.chain_types, topology.chain_nums):
        chain_pdb = _generate_temporary_prefix() + ".pdb"
        chain_pdbs.append(chain_pdb)
        chain_type.to_pdb(chain_pdb)
        packmol_str += f"structure {chain_pdb}\n"
        packmol_str += f"  number {chain_num}\n"
        packmol_str += f"  inside box {box}\n"
        for instruction in chain_type.packmol_instructions:
            packmol_str += f"  {instruction}\n"
        packmol_str += "end structure\n"

    # save packmol string to input file
    packmol_input = _generate_temporary_prefix() + ".inp"
    open(packmol_input, 'w').write(packmol_str)

    # run packmol while suppressing output
    os.system(f"packmol < {packmol_input} > /dev/null")

    # get positions from pdb
    t = md.load(output_pdb)
    positions = t.xyz[0]

    # remove all temporary files
    os.system(f"rm {output_pdb}")
    os.system(f"rm {packmol_input}")
    for c in chain_pdbs:
        os.system(f"rm {c}")

    return positions


def initialize_velocities_from_temperature(n_beads, temperature):
    """Returns an initial random velocity set.

    Parameters
    ----------
    n_beads : int
        Number of particles.
    temperature : float
        Target temperature

    Returns
    -------
    velocities : np.ndarray, shape=(num_atoms, 3)
    """
    # randomly initialize velocities
    velocities = np.random.rand(n_beads, 3)

    # scale velocities to target temperature
    velocities = utils.rescale_velocities(velocities, temperature)

    return velocities


class State(object):
    """Container object that stores information about the current state of a
    simulation.

    Attributes
    ----------
    positions : np.ndarray, shape=(num_particles, 3)
        Current positions of all particles.
    velocities : np.ndarray, shape=(num_particles, 3)
        Current velocities of all particles.
    forces : np.ndarray, shape=(num_particles, 3)
        Current force acting on all particles.
    potential_energy : float
        Current potential energy of the system.
    kinetic_energy : float
        Current kinetic energy of the system.
    """

    def __init__(
            self, steps=0, positions=None, velocities=None, forces=None,
            potential_energy=None, kinetic_energy=None
    ):
        self.steps = steps
        self.positions = positions
        self.velocities = velocities
        self.forces = forces
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy


class Simulation(object):

    def __init__(self, system, integrator):
        # save system and integrator to attributes
        self.system = system
        self.integrator = integrator
        self.integrator.simulation = self

        # initialize attributes for thermo reporting
        self.thermo_frequency = 1000
        self._thermo_verbose = False
        self._thermo_file = None

        # initialize attributes for traj reporting
        self.traj_frequency = 1000
        self._traj_file = None
        self.traj_min_image = True
        self._current_model_index = 1

        # initialize state
        self.state = State()

    def initialize(self, preserve_state=False):
        # compile the system
        self.system.initialize()

        # initialize state if not preserving current state
        if not preserve_state:
            # set positions
            positions = initialize_positions(self.system.topology)
            self.set_positions(positions)

            # set velocities
            velocities = np.zeros_like(positions)
            self.set_velocities(velocities)
            self.state.steps = 0
        else:
            # reset positions to recalculate forces and potential energy
            positions = self.state.positions
            self.set_positions(positions)

    def set_positions(self, positions):
        # check that number of positions is correct
        if self.system.topology.n_beads != len(positions):
            raise ValueError(
                "number of positions doesn't match the number of beads in the topology"
            )

        # set positions and compute forces and potential energy
        self.state.positions = positions
        forces, potential_energy = self.system.force_energy_function(positions)
        self.state.forces = forces
        self.state.potential_energy = potential_energy

    def set_velocities(self, velocities):
        # check that number of velocities is correct
        if self.system.topology.n_beads != len(velocities):
            raise ValueError(
                "number of velocities doesn't match the number of beads in the topology"
            )

        # set velocities and compute kinetic energy
        self.state.velocities = velocities
        self.state.kinetic_energy = utils.compute_kinetic_energy(velocities)

    def set_velocities_to_temperature(self, temperature=1.0):
        velocities = initialize_velocities_from_temperature(self.system.topology.n_beads, temperature)
        self.set_velocities(velocities)

    def minimize_energy(self):
        force_energy_function = self.system.force_energy_function
        potential_energy, positions = conjugate_gradient(force_energy_function, self.state.positions)
        forces, _ = force_energy_function(positions)
        self.state.positions = positions
        self.state.forces = forces
        self.state.potential_energy = potential_energy

    def step(self, steps):
        steps_elapsed = 0
        start_time = time.time()
        min_frequency = min(self.thermo_frequency, self.traj_frequency)
        while steps > 0:
            if steps > min_frequency:
                steps_to_take = min_frequency
            else:
                steps_to_take = steps
            state = self.integrator.step(steps_to_take)
            steps -= steps_to_take
            steps_elapsed += steps_to_take
            self.state = state

            # report
            if steps_elapsed % self.thermo_frequency == 0:
                self._thermo_update(time.time() - start_time)
            if steps_elapsed % self.traj_frequency == 0:
                self._traj_update()

    @property
    def thermo_file(self):
        return self._thermo_file

    @thermo_file.setter
    def thermo_file(self, filename):
        self._thermo_file = filename
        header = self._create_thermo_header()
        if self.thermo_verbose:
            print(header)
        if self._thermo_file is not None:
            with open(self._thermo_file, 'w') as f:
                f.write(self._create_thermo_header() + "\n")

    @property
    def thermo_verbose(self):
        return self._thermo_verbose

    @thermo_verbose.setter
    def thermo_verbose(self, value):
        if value:
            print(self._create_thermo_header())
        self._thermo_verbose = value

    @staticmethod
    def _create_thermo_header():
        header = '#"Steps","Potential Energy","Kinetic Energy","Temperature","Elapsed Time (s)"'
        return header

    def _thermo_update(self, elapsed_time):
        steps = self.state.steps
        potential_energy = self.state.potential_energy
        kinetic_energy = self.state.kinetic_energy
        temperature = utils.compute_instantaneous_temperature(self.state.velocities)
        line = f"{steps},{potential_energy},{kinetic_energy},{temperature},{elapsed_time}"
        if self.thermo_verbose:
            print(line)
        if self._thermo_file is not None:
            with open(self._thermo_file, 'a') as f:
                f.write(line + "\n")

    @property
    def traj_file(self):
        return self._traj_file

    @traj_file.setter
    def traj_file(self, filename):
        self._traj_file = filename
        self._current_model_index = 1

    def _traj_update(self):
        if self._traj_file is not None:
            self.system.topology.to_pdb(self._traj_file,
                                        model_index=self._current_model_index,
                                        positions=self.state.positions,
                                        append=self._current_model_index != 1,
                                        min_image=self.traj_min_image)
            self._current_model_index += 1
