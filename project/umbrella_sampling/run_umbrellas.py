import os.path

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


# parameters
copolymer_sequence = "AAAAABBBBB"
eps_wall = 1.0
k_bias = 1.0
umbrellas = np.arange(2, 20, 2, dtype=float)
box_lengths = np.array([50., 10., 10.])

# ================= #
# CREATE SIMULATION #
# ================= #

# create bead types
bead_type_a = BeadType("A")
bead_type_b = BeadType("B")
beads = [Bead(b, BeadType.get_by_name(b)) for b in copolymer_sequence]
copolymer = LinearChain("copolymer", beads)
copolymer.packmol_instructions.append("center")
copolymer.packmol_instructions.append(f"fixed {box_lengths[0]*5} {box_lengths[1]*5} {box_lengths[2]*5} 0. 0. 0.")

# create topology
topology = Topology()
topology.add_chain(copolymer, n=1)
topology.box_lengths = box_lengths

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
harmonic_bias.add_interaction(group, k=0.5, r0=5.0, axis=0)

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

# loop through each umbrella
for r0 in umbrellas:
    # create copolymer directory if it doesn't exist
    if not os.path.exists(copolymer_sequence):
        os.mkdir(copolymer_sequence)

    # prefix of output files
    file_prefix = os.path.join(copolymer_sequence, f"x{r0}")

    # traj output file
    traj_file = file_prefix + "_traj.pdb"
    if os.path.exists(traj_file):
        print(f"{traj_file} already exists...")
        continue

    # set bias strength and compile simulation
    harmonic_bias.set_parameter_value('r0', r0, 0)
    simulation.initialize()

    # minimize energy
    simulation.minimize_energy()

    # set up reporting
    simulation.thermo_file = None
    simulation.thermo_frequency = 100000
    simulation.thermo_verbose = True
    simulation.traj_file = traj_file
    simulation.traj_frequency = 1000
    simulation.traj_min_image = True

    # run
    simulation.step(10000000)
