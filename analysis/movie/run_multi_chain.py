import numpy as np

from mdlib import *


def run_multi_chain(eps_wall, sequence, k_bias, umbrellas):
    # # create run directory
    # run_dir = create_run_dir(eps_wall, sequence, k_bias, x0)

    # # traj output file
    # traj_index = 0
    # while True:
    #     traj_file = os.path.join(run_dir, f"traj{traj_index}.pdb")
    #     if os.path.exists(traj_file):
    #         traj_index += 1
    #     else:
    #         break

    # create topology
    topology = Topology()
    topology.box_lengths = [50, 10, 10]
    topology.periodicity[:] = False

    # initialize potentials
    ljwca = LJWCA()
    harmonic_bond = HarmonicBond()
    wall_ljwca = WallLJWCA()
    harmonic_bias = HarmonicBias()

    for n, u in enumerate(umbrellas):
        bead_name_a = f"A{n}"
        bead_name_b = f"B{n}"

        # create bead types
        if not BeadType.has_bead_type(bead_name_a):
            BeadType(bead_name_a)
        if not BeadType.has_bead_type(bead_name_b):
            BeadType(bead_name_b)

        # create copolymer
        beads = [Bead(b, BeadType.get_by_name(f"{b}{n}")) for b in sequence]
        copolymer = LinearChain("copolymer", beads)

        # set packing instructions for copolymer
        copolymer.packmol_instructions = []
        copolymer.packmol_instructions.append("center")
        copolymer.packmol_instructions.append(f"fixed {u*10.} {topology.box_lengths[1]*5} {topology.box_lengths[2]*5} 0. 0. 0.")

        # add chain to topology
        topology.add_chain(copolymer, n=1)

        # create LJ potential
        ljwca.add_interaction(bead_name_a, bead_name_a, lambda_wca=1.0)
        ljwca.add_interaction(bead_name_a, bead_name_b, lambda_wca=1.0)
        ljwca.add_interaction(bead_name_b, bead_name_b, lambda_wca=1.0)

        # create harmonic potentials
        harmonic_bond.add_interaction(bead_name_a, bead_name_a)
        harmonic_bond.add_interaction(bead_name_a, bead_name_b)
        harmonic_bond.add_interaction(bead_name_b, bead_name_b)

        # create wall interactions
        wall_ljwca.add_interaction(bead_name_a, eps=eps_wall, cut=7.5, upper_bound=topology.box_lengths[0])
        wall_ljwca.add_interaction(bead_name_b, eps=eps_wall, upper_bound=topology.box_lengths[0], lambda_lj=0.0, lambda_wca=1.0)

        # create harmonic bias
        group = np.array([b.index for b in list(topology.chains)[n].beads])
        harmonic_bias.add_interaction(group, k=k_bias, r0=u, axis=0)

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
    simulation.initialize()

    # minimize energy
    simulation.minimize_energy()

    # set up reporting
    simulation.thermo_file = None
    simulation.thermo_frequency = 100000
    simulation.thermo_verbose = True
    simulation.traj_file = "traj.pdb"
    simulation.traj_frequency = 1000
    simulation.traj_min_image = True

    # step
    simulation.step(1000000)


if __name__ == '__main__':
    run_multi_chain(1.0, "AAAAABBBBB", 5.0, umbrellas=np.arange(2, 20, 4))
