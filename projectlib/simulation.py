"""simulation.py: Contains classes for running simulations and storing their
states.
"""
__all__ = ['State', 'Simulation']

import time

import numpy as np

from projectlib import utils


def initialize_positions(num_atoms, box_length):
    """Returns an array of initial positions of each atom, placed on a cubic
    lattice for convenience.

    Parameters
    ----------
    num_atoms : int
        Number of particles.
    box_length : float
        Length of each side of the box

    Returns
    -------
    positions : np.ndarray, shape=(num_atoms, 3)
        Initialized positions
    """
    # initialize position array
    positions = np.zeros((num_atoms, 3), dtype=float)

    # compute integer grid # of locations for cubic lattice
    NLat = int(num_atoms ** (1. / 3.) + 1.)
    LatSpac = box_length / NLat
    #make an array of lattice sites
    r = LatSpac * np.arange(NLat, dtype=float) - 0.5 * box_length
    #loop through x, y, z positions in lattice until done
    #for every atom in the system
    i = 0
    for x in r:
        for y in r:
            for z in r:
                positions[i] = np.array([x,y,z], float)
                #add a random offset to help initial minimization
                Offset = 0.1 * LatSpac * (np.random.rand(3) - 0.5)
                positions[i] = positions[i] + Offset
                i += 1
                #if done placing atoms, return
                if i >= num_atoms:
                    return positions
    return positions


def initialize_velocities_from_temperature(num_atoms, temperature):
    """Returns an initial random velocity set.

    Parameters
    ----------
    num_atoms : int
        Number of particles.
    temperature : float
        Target temperature

    Returns
    -------
    velocities : np.ndarray, shape=(num_atoms, 3)
    """
    # randomly initialize velocities
    velocities = np.random.rand(num_atoms, 3)

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

    def __init__(self, positions, velocities, forces, potential_energy, kinetic_energy):
        self.positions = positions
        self.velocities = velocities
        self.forces = forces
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy


class Simulation(object):

    def __init__(self, system, integrator):
        self.system = system
        self.integrator = integrator
        self.integrator.simulation = self
        self.state = None

    def initialize(self):
        positions = initialize_positions(self.system.n_beads, self.system.box_lengths[0])
        velocities = initialize_velocities_from_temperature(self.system.n_beads, 1.0)
        forces, potential_energy = self.system.force_energy_function(positions)
        kinetic_energy = utils.compute_kinetic_energy(velocities)
        self.state = State(positions, velocities, forces, potential_energy, kinetic_energy)

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
