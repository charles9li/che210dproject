"""simulation.py: Contains classes for running simulations and storing their
states.
"""
__all__ = ['State', 'Simulation']

import os
import sys
import time
import random
import string

import numpy as np

import mdtraj as md

from projectlib.minimize_energy import conjugate_gradient
from projectlib import utils


def _generate_temporary_prefix():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))


def initialize_positions(topology, edge_buffer=0.2):
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

    def __init__(self, steps, positions, velocities, forces, potential_energy, kinetic_energy):
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

        # initialize attributes for reporting
        self.thermo_interval = 1000
        self._thermo_file = None
        self._thermo_append = False
        self.traj_interval = 1000
        self._traj_file = None
        self._traj_append = False

        # initialize state
        self.state = None

    def initialize(self):
        self.system.initialize()
        # positions = initialize_positions(self.system.topology.n_beads, self.system.topology.box_lengths[0])
        positions = initialize_positions(self.system.topology)
        velocities = initialize_velocities_from_temperature(self.system.topology.n_beads, 1.0)
        # velocities = np.zeros_like(positions)
        forces, potential_energy = self.system.force_energy_function(positions)
        kinetic_energy = utils.compute_kinetic_energy(velocities)
        self.state = State(0, positions, velocities, forces, potential_energy, kinetic_energy)

    def minimize_energy(self):
        force_energy_function = self.system.force_energy_function
        potential_energy, positions = conjugate_gradient(force_energy_function, self.state.positions)
        forces, _ = force_energy_function(positions)
        self.state.positions = positions
        self.state.forces = forces
        self.state.potential_energy = potential_energy

    def step(self, steps):
        steps_elapsed = 0
        thermo_frequency = 1000
        start_time = time.time()
        while steps > 0:
            if steps > thermo_frequency:
                steps_to_take = thermo_frequency
            else:
                steps_to_take = steps
            state = self.integrator.step(steps_to_take)
            steps -= steps_to_take
            steps_elapsed += steps_to_take
            self.state = state
            temperature = utils.compute_instantaneous_temperature(state.velocities)
            print(steps_elapsed, state.potential_energy, state.kinetic_energy, temperature, time.time() - start_time)
