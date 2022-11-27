"""utils.py: Contains various utility functions."""
__all__ = ['compute_kinetic_energy', 'compute_instantaneous_temperature', 'rescale_velocities']

import numpy as np


def compute_kinetic_energy(velocities):
    """Returns the kinetic energy computed using the velocities.

    Parameters
    ----------
    velocities : np.ndarray, shape=(num_atoms, 3)
        Velocities of each particle in the system.

    Returns
    -------
    kinetic_energy : float
        Kinetic energy.
    """
    return 0.5 * np.sum(velocities * velocities)


def compute_instantaneous_temperature(velocities):
    """Returns the instantaneous temperature computed using the velocities.

    Parameters
    ----------
    velocities : np.ndarray, shape=(num_atoms, 3)
        Velocities of each particle in the system.

    Returns
    -------
    temperature : float
        Instantaneous temperature.
    """
    return 2 * compute_kinetic_energy(velocities) / (3 * len(velocities))


def rescale_velocities(velocities, temperature):
    """Rescales velocities in the system to the target temperature.

    Parameters
    ----------
    velocities : np.ndarray, shape=(num_atoms, 3)
        Velocities of each particle in the system.
    temperature : float
        Target temperature.

    Returns
    -------
    velocities : np.ndarray, shape=(num_atoms, 3)
        Rescaled velocities.
    """
    # recenter to zero net momentum (assuming all masses same)
    velocities = velocities - velocities.mean(axis=0)

    # find the total kinetic energy
    kinetic_energy = compute_kinetic_energy(velocities)

    # find velocity scale factor from ratios of kinetic energy
    scale_factor = np.sqrt(1.5 * len(velocities) * temperature / kinetic_energy)
    velocities = velocities * scale_factor

    return velocities
