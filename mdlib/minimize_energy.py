__all__ = ['conjugate_gradient']

import time

import numpy as np


def line_search(force_energy_function, positions, search_dir, dx=0.001,
                energy_frac_tol=1e-8, accel=1.5, max_inc=10.0, max_iter=10000):
    """Performs a line search along a specified direction.

    Parameters
    ----------
    force_energy_function : callable
        Function that takes in positions and returns the potential energy.
    positions : array_like, shape=(N,3)
        Starting positions.
    search_dir : array_like, shape=(N,3)
        Array of gradient directions.
    dx : float
        Initialize step size.
    energy_frac_tol : float, default=1e-8
        Fractional energy tolerance.
    accel : float, default=1.5
        Acceleration factor.
    max_inc : float, default=10.0
        The maximum increase in energy for bracketing.
    max_iter : int, default=10000
        maximum number of iteration steps.

    Returns
    -------
    PEnergy: value of potential energy at minimum along Dir
    Pos: minimum energy (N,3) position array along Dir
    """
    # start the iteration counter
    iteration_counter = 0

    # find the normalized direction
    norm_dir = np.clip(search_dir, -1.e100, 1.e100)
    norm_dir = norm_dir / np.sqrt(np.sum(norm_dir * norm_dir))

    # take the first two steps and compute energies
    dists = [0., dx]
    PEs = [force_energy_function(positions + norm_dir * x)[1] for x in dists]

    # if the second point is not downhill in energy, back off and take a
    # shorter step until we find one
    while PEs[1] > PEs[0]:
        iteration_counter += 1
        dx = dx * 0.5
        dists[1] = dx
        _, PEs[1] = force_energy_function(positions + norm_dir * dx)

    # find a third point
    dists = dists + [2. * dx]
    PEs = PEs + [force_energy_function(positions + norm_dir * 2. * dx)[1]]

    # keep stepping forward until the third point is higher in energy; then we
    # have bracketed a minimum
    while PEs[2] < PEs[1]:
        # update iteration counter
        iteration_counter += 1

        # find a fourth point and evaluate energy
        dists = dists + [dists[-1] + dx]
        PEs = PEs + [force_energy_function(positions + norm_dir * dists[-1])[1]]

        # check if we increased too much in energy; if so, back off
        if (PEs[3] - PEs[0]) > max_inc * (PEs[0] - PEs[2]):
            PEs = PEs[:3]
            dists = dists[:3]
            dx = dx * 0.5
        else:
            # we found a good energy; shift all the points over
            PEs = PEs[-3:]
            dists = dists[-3:]
            dx = dx * accel

    # we've bracketed a minimum; now we want to find it to high accuracy
    OldPE3 = 1.e300
    # loop over successive narrowing of the distance range
    while True:
        # update iteration counter
        iteration_counter += 1

        # stop if we reached max number of iterations
        if iteration_counter > max_iter:
            print("Warning: maximum number of iterations reached in line search.")
            break

        # unpack distances for ease of code-reading
        d0, d1, d2 = dists
        PE0, PE1, PE2 = PEs

        # use a parobolic approximation to estimate the minimum location
        d10 = d0 - d1
        d12 = d2 - d1
        numerator = d12*d12*(PE0-PE1) - d10*d10*(PE2-PE1)
        denominator = d12*(PE0-PE1) - d10*(PE2-PE1)
        if denominator == 0:
            # parabolic extrapolation won't work; set new dist = 0
            d3 = 0
        else:
            # location of parabolic minimum
            d3 = d1 + 0.5 * numerator / denominator

        # compute the new potential energy
        _, PE3 = force_energy_function(positions + norm_dir * d3)

        # sometimes the parabolic approximation can fail;
        # check if d3 is out of range < d0 or > d2 or the new energy is higher
        if d3 < d0 or d3 > d2 or PE3 > PE0 or PE3 > PE1 or PE3 > PE2:
            # instead, just compute the new distance by bisecting two
            # of the existing points along the line search
            if abs(d2 - d1) > abs(d0 - d1):
                d3 = 0.5 * (d2 + d1)
            else:
                d3 = 0.5 * (d0 + d1)
            _, PE3 = force_energy_function(positions + norm_dir * d3)

        # decide which three points to keep; we want to keep
        # the three that are closest to the minimum
        if d3 < d1:
            if PE3 < PE1:
                # get rid of point 2
                dists, PEs = [d0, d3, d1], [PE0, PE3, PE1]
            else:
                # get rid of point 0
                dists, PEs = [d3, d1, d2], [PE3, PE1, PE2]
        else:
            if PE3 < PE1:
                # get rid of point 0
                dists, PEs = [d1, d3, d2], [PE1, PE3, PE2]
            else:
                # get rid of point 2
                dists, PEs = [d0, d1, d3], [PE0, PE1, PE3]

        # check how much we've changed
        if abs(OldPE3 - PE3) < energy_frac_tol * abs(PE3):
            # the fractional change is less than the tolerance,
            # so we are done and can exit the loop
            break
        OldPE3 = PE3

    # return the position array at the minimum (point 1)
    positions_min = positions + norm_dir * dists[1]
    PEMin = PEs[1]

    return PEMin, positions_min


def conjugate_gradient(force_energy_function, positions, dx=0.001, energy_frac_tol_ls=1e-8, energy_frac_tol_cg=1e-10):
    """Performs a conjugate gradient search.

    Parameters
    ----------
    force_energy_function : callable
        Function that takes in positions and returns forces and the potential
        energy.
    positions : array_like, shape=(N,3)
        Starting positions.
    dx : float, default=0.001
        Initial step size.
    energy_frac_tol_ls : float, default=1e-8
        Fractional energy tolerance for line search.
    energy_frac_tol_cg : float, default=1d-10
        Fractional energy tolerance for conjugate gradient.

    Returns
    -------
    potential_energy : float
        Value of potential energy at minimum.
    positions : array_like, shape=(N,3)
        Minimum energy (N,3) position array.
    """
    # initialize variables
    forces, PE = force_energy_function(positions)

    # initial direction is the steepest descent, i.e., along the forces
    direction = forces
    old_PE = 1.e300

    # variables to keep track of the progress
    i = 0
    last_time = time.time()

    # loop until change in energy is less than a tolerance
    while abs(PE - old_PE) > energy_frac_tol_cg * abs(PE):

        old_PE = PE
        i += 1

        # do a line search along the current direction
        PE, positions = line_search(force_energy_function, positions, direction,
                                    dx=dx, energy_frac_tol=energy_frac_tol_ls)

        # save the old forces and get the new ones
        old_forces = forces
        forces, PE = force_energy_function(positions)

        # use the conjugate gradient update to find the new direction
        gamma = np.sum((forces - old_forces) * forces) / np.sum(old_forces * old_forces)
        direction = forces + gamma * direction

        # print out the current results at fixed intervals
        if time.time() > last_time + 1.0:
            print("Current energy for round %i (force magnitude): %.5f  (%7.1e)" % (i, PE, np.sum(forces * forces)))
            last_time = time.time()

    return PE, positions
