"""integrators.py: Contains various integrators for performing molecular
dynamics simulations for Lennard-Jones systems.
"""
__all__ = ['VelocityVerletIntegrator', 'LangevinIntegrator', 'LangevinIntegratorLAMMPS']

import numpy as np

from numba import jit

import projectlib


@jit(nopython=True)
def _step_integrator(
        force_energy_function,
        positions, velocities, forces,
        step_size, *args
):
    """Numba-accelerated helper function used to take a time step. Every
    integrator will need to implement this.

    Parameters
    ----------
    force_energy_function : callable
        Numba-accelerated function that takes in positions and calculates
        forces and potential energies.
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    step_size : float
        Size of time step.

    Returns
    -------
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    potential_energy : float
        Potential energy of the system after taking time step.
    """
    return np.zeros_like(positions), np.zeros_like(positions), np.zeros_like(positions), 0.0


class _Integrator(object):
    """Base class that all integrators inherit from.

    Attributes
    ----------
    step_size : int, default=0.002
        Time step used for integration.
    simulation : projectlib.Simulation
        Used to run a simulation. This attribute will be set when this
        integrator is added to a simulation.
    """
    STEP_FUNCTION = staticmethod(_step_integrator)

    def __init__(self, step_size=0.002):
        self.step_size = step_size
        self.simulation = None

    def _additional_step_function_args(self):
        """Additional arguments to pass to the step function if necessary."""
        return ()

    def step(self, steps):
        """Take a specified number of time steps.

        Parameters
        ----------
        steps : int
            Number of time steps to take.
        """
        # get current state of simulation
        state = self.simulation.state
        curr_steps = state.steps
        positions = state.positions
        velocities = state.velocities
        forces = state.forces
        potential_energy = state.potential_energy

        # get function used to compute forces and potential energy
        force_function = self.simulation.system.force_energy_function

        # get additional arguments
        additional_args = self._additional_step_function_args()

        # take steps
        for i in range(steps):
            positions, velocities, forces, potential_energy = self.STEP_FUNCTION(force_function, positions, velocities, forces, self.step_size, *additional_args)

        # return state
        kinetic_energy = projectlib.utils.compute_kinetic_energy(velocities)
        return projectlib.State(curr_steps + steps, positions, velocities, forces, potential_energy, kinetic_energy)


@jit(nopython=True)
def _step_velocity_verlet(
        force_energy_function,
        positions, velocities, forces,
        step_size
):
    """Take a time step using the velocity-Verlet algorithm.

    Parameters
    ----------
    force_energy_function : callable
        Numba-accelerated function that takes in positions and calculates
        forces and potential energies.
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    step_size : float
        Size of time step.

    Returns
    -------
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    potential_energy : float
        Potential energy of the system after taking time step.
    """
    # precompute half of time step
    half_step_size = 0.5 * step_size

    # take first half step in velocity
    velocities = velocities + half_step_size * forces

    # update positions and forces
    positions = positions + step_size * velocities
    forces, potential_energy = force_energy_function(positions)

    # take second half step in velocity
    velocities = velocities + half_step_size * forces

    return positions, velocities, forces, potential_energy


class VelocityVerletIntegrator(_Integrator):
    """Velocity-Verlet integrator."""
    STEP_FUNCTION = staticmethod(_step_velocity_verlet)


@jit(nopython=True)
def _step_langevin(
        force_energy_function,
        positions, velocities, forces,
        step_size,
        temperature, friction_coefficient
):
    """Take a time step using the velocity-Verlet algorithm. Uses the algorithm
    from Bussi and Parrinello.

    Parameters
    ----------
    force_energy_function : callable
        Numba-accelerated function that takes in positions and calculates
        forces and potential energies.
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    step_size : float
        Size of time step.
    temperature : float
        Temperature of the heat bath.
    friction_coefficient : float
        Determines how strongly the system is coupled to the heat bath.

    Returns
    -------
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    potential_energy : float
        Potential energy of the system after taking time step.

    References
    ----------
    [1] Bussi, G.; Parrinello, M. Accurate Sampling Using Langevin Dynamics.
    Phys. Rev. E 2007, 75 (5), 056707.
    https://doi.org/10.1103/PhysRevE.75.056707.
    """
    # precompute coefficients
    c1 = np.exp(-0.5 * friction_coefficient * step_size)
    c2 = np.sqrt((1 - c1 * c1) * temperature)

    # propagate velocity and positions
    velocities = c1 * velocities + c2 * np.random.randn(*positions.shape)
    positions = positions + velocities * step_size + 0.5 * step_size * step_size * forces
    velocities = velocities + 0.5 * step_size * forces

    # compute new forces
    forces, potential_energy = force_energy_function(positions)

    # finish propagation of velocities
    velocities = velocities + 0.5 * step_size * forces
    velocities = c1 * velocities + c2 * np.random.randn(*positions.shape)

    return positions, velocities, forces, potential_energy


class LangevinIntegrator(_Integrator):
    """Langevin integrator that uses the algorithm from Bussi and Parrinello.

    Attributes
    ----------
    temperature : float, default=1.0
        Temperature of the heat bath.
    friction_coefficient : float, default=1.0
        Determines how strongly the system is coupled to the heat bath.

    References
    ----------
    [1] Bussi, G.; Parrinello, M. Accurate Sampling Using Langevin Dynamics.
    Phys. Rev. E 2007, 75 (5), 056707.
    https://doi.org/10.1103/PhysRevE.75.056707.
    """
    STEP_FUNCTION = staticmethod(_step_langevin)

    def __init__(self, step_size=0.002, temperature=1.0, friction_coefficient=1.0):
        super(LangevinIntegrator, self).__init__(step_size)
        self.temperature = temperature
        self.friction_coefficient = friction_coefficient

    def _additional_step_function_args(self):
        """The temperature and friction coefficient are additional arguments
        passed to the step function.
        """
        return self.temperature, self.friction_coefficient


@jit(nopython=True)
def _step_langevin_lammps(
        force_energy_function,
        positions, velocities, forces,
        step_size,
        temperature, friction_coefficient
):
    """Take a time step using the velocity-Verlet algorithm. Uses the algorithm
    from LAMMPS.

    Parameters
    ----------
    force_energy_function : callable
        Numba-accelerated function that takes in positions and calculates
        forces and potential energies.
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    step_size : float
        Size of time step.
    temperature : float
        Temperature of the heat bath.
    friction_coefficient : float
        Determines how strongly the system is coupled to the heat bath.

    Returns
    -------
    positions : np.ndarray, shape=(n_particles, 3)
        Positions of each particle.
    velocities : np.ndarray, shape=(n_particles, 3)
        Velocities of each particle.
    forces : np.ndarray, shape=(n_particles, 3)
        Forces acting on each particle.
    potential_energy : float
        Potential energy of the system after taking time step.
    """
    # precompute half of step size and coefficient for random noise term
    half_step_size = 0.5 * step_size
    random_coefficient = np.sqrt(step_size * temperature * friction_coefficient)

    # first velocity half update
    random_noise = random_coefficient * np.random.randn(*positions.shape)
    velocities = velocities + half_step_size * (forces - friction_coefficient * velocities) + random_noise

    # update positions and forces
    positions = positions + velocities * step_size
    forces, potential_energy = force_energy_function(positions)

    # second velocity half update
    random_noise = random_coefficient * np.random.randn(*positions.shape)
    velocities = velocities + half_step_size * (forces - friction_coefficient * velocities) + random_noise

    return positions, velocities, forces, potential_energy


class LangevinIntegratorLAMMPS(LangevinIntegrator):
    """Langevin integrator that uses the algorithm from LAMMPS."""
    STEP_FUNCTION = staticmethod(_step_langevin_lammps)
