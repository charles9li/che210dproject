import os

import numpy as np

from mdlib import (
    Bead,
    BeadType,
    ForceField,
    HarmonicBias,
    HarmonicBond,
    LangevinIntegrator,
    LinearChain,
    LJWCA,
    Simulation,
    System,
    Topology,
    WallLJWCA
)

from utils import create_run_dir, find_uncorrelated_samples


def run_umbrella(eps_wall, sequence, k_bias, x0, steps=10000000, verbose=False, keep_traj=False):
    # create run directory
    run_dir = create_run_dir(eps_wall, sequence, k_bias, x0)

    # determine traj index
    traj_index = 0
    while True:
        x_filename = os.path.join(run_dir, f"xsamples{traj_index}.txt")
        if os.path.exists(x_filename):
            traj_index += 1
        else:
            break

    # traj output file
    traj_file = os.path.join(run_dir, f"traj{traj_index}.pdb")

    # create bead types
    if not BeadType.has_bead_type("A"):
        BeadType("A")
    if not BeadType.has_bead_type("B"):
        BeadType("B")

    # create copolymer
    beads = [Bead(b, BeadType.get_by_name(b)) for b in sequence]
    copolymer = LinearChain("copolymer", beads)

    # create topology
    topology = Topology()
    topology.add_chain(copolymer, n=1)
    topology.box_lengths = [50, 10, 10]
    topology.periodicity[:] = False

    # create LJ potential
    ljwca = LJWCA()
    ljwca.add_interaction("A", "A", lambda_wca=1.0)
    ljwca.add_interaction("A", "B", lambda_wca=1.0)
    ljwca.add_interaction("B", "B", lambda_wca=1.0)

    # create harmonic potentials
    harmonic_bond = HarmonicBond()
    harmonic_bond.add_interaction("A", "A")
    harmonic_bond.add_interaction("A", "B")
    harmonic_bond.add_interaction("B", "B")

    # create wall interactions
    wall_ljwca = WallLJWCA()
    wall_ljwca.add_interaction("A", eps=eps_wall, cut=7.5, upper_bound=topology.box_lengths[0])
    wall_ljwca.add_interaction("B", eps=eps_wall, upper_bound=topology.box_lengths[0], lambda_lj=0.0, lambda_wca=1.0)

    # create harmonic bias
    harmonic_bias = HarmonicBias()
    group = np.array([b.index for b in list(topology.chains)[0].beads])
    harmonic_bias.add_interaction(group, k=k_bias, r0=5.0, axis=0)

    # create force field and system
    force_field = ForceField()
    force_field.add_potential(ljwca)
    force_field.add_potential(harmonic_bond)
    force_field.add_potential(harmonic_bias)
    force_field.add_potential(wall_ljwca)
    system = System(topology, force_field)

    # create integrator
    integrator = LangevinIntegrator(step_size=0.002)

    # create simulation and initialize
    simulation = Simulation(system, integrator)

    # set packing instructions for copolymer
    copolymer.packmol_instructions = []
    copolymer.packmol_instructions.append("center")
    copolymer.packmol_instructions.append(f"fixed {x0*10} {topology.box_lengths[1]*5} {topology.box_lengths[2]*5} 0. 0. 0.")

    # set bias strength and compile simulation
    harmonic_bias.set_parameter_value('r0', x0, 0)
    simulation.initialize()

    # minimize energy
    simulation.minimize_energy()

    # set up reporting
    simulation.thermo_file = None
    simulation.thermo_frequency = 100000
    simulation.thermo_verbose = verbose
    simulation.traj_file = traj_file
    simulation.traj_frequency = 1000
    simulation.traj_min_image = True

    # run
    print(f"Begin running {traj_file} ...")
    simulation.step(steps)
    print(f"Finished running {traj_file} ...")

    # get uncorrelated samples for distance from wall and save to file
    x_n = find_uncorrelated_samples(traj_file)
    print(f"Finding uncorrelated samples and saving to {x_filename} ...")
    np.savetxt(x_filename, x_n)

    # delete traj if specified
    if not keep_traj:
        print(f"Deleting {traj_file} ...")
        os.system(f"rm {traj_file}")


if __name__ == '__main__':
    run_umbrella(1.0, "AAAAABBBBB", 5.0, 1.0)
