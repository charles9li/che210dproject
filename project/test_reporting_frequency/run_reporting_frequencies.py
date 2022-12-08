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


# ================= #
# CREATE SIMULATION #
# ================= #

# create block copolymer
bead_type_a = BeadType("A")
bead_type_b = BeadType("B")
beads = [Bead("A", bead_type_a)]*5 + [Bead("B", bead_type_b)]*5
block_copolymer = LinearChain("block_copolymer", beads)
block_copolymer.packmol_instructions.append("center")
block_copolymer.packmol_instructions.append("fixed 250. 50. 50. 0. 0. 0.")

# create topology
topology = Topology()
topology.add_chain(block_copolymer, n=1)
topology.box_lengths = np.array([50., 10., 10.])

# create LJ potential
lj = LJWCA()
lj.add_interaction("A", "A", lambda_wca=1.0)
lj.add_interaction("A", "B", lambda_wca=1.0)
lj.add_interaction("B", "B", lambda_wca=1.0)

# create harmonic potentials
harmonic_bond = HarmonicBond()
harmonic_bond.add_interaction("A", "A")
harmonic_bond.add_interaction("A", "B")
harmonic_bond.add_interaction("B", "B")

# # create wall interactions
# wall_ljwca = WallLJWCA()
# wall_ljwca.add_interaction("A", cut=7.5, upper_bound=topology.box_lengths[0])
# wall_ljwca.add_interaction("B", upper_bound=topology.box_lengths[0], lambda_lj=0.0, lambda_wca=1.0)

# create harmonic bias
harmonic_bias = HarmonicBias()
group = np.array([b.index for b in list(topology.chains)[0].beads])
harmonic_bias.add_interaction(group, k=0.5, r0=55.0, axis=0)

# create force field and system
force_field = ForceField()
force_field.add_potential(lj)
force_field.add_potential(harmonic_bond)
force_field.add_potential(harmonic_bias)
# force_field.add_potential(wall_ljwca)
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator(step_size=0.002)

# create simulation and initialize
simulation = Simulation(system, integrator)

# loop through different bias strengths
for report_frequency in [100, 500, 1000, 5000, 10000]:

    # reinitialize simulation
    simulation.initialize()

    # minimize energy
    simulation.minimize_energy()

    # prefix of output files
    file_prefix = f"rf{report_frequency}"

    # set up reporting
    simulation.thermo_file = None
    simulation.thermo_frequency = 10000
    simulation.thermo_verbose = True
    simulation.traj_file = file_prefix + "_traj.pdb"
    simulation.traj_frequency = report_frequency
    simulation.traj_min_image = True

    # run
    simulation.step(1000000)
