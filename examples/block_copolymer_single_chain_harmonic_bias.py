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
    Topology
)

file_prefix = "block_copolymer_single_chain_harmonic_bias"

# create block copolymer
bead_type_a = BeadType("A")
bead_type_b = BeadType("B")
beads = [Bead("A", bead_type_a)]*5 + [Bead("B", bead_type_b)]*5
block_copolymer = LinearChain("block_copolymer", beads)
block_copolymer.packmol_instructions.append("center")
block_copolymer.packmol_instructions.append("fixed 200. 50. 50. 0. 0. 0.")

# create topology
topology = Topology()
topology.add_chain(block_copolymer, n=1)
topology.box_lengths = np.array([40., 10., 10.])

# create LJ potential
lj = LJWCA()
lj.add_interaction("A", "A")
lj.add_interaction("A", "B")
lj.add_interaction("B", "B")

# create harmonic potentials
harmonic_bond = HarmonicBond()
harmonic_bond.add_interaction("A", "A")
harmonic_bond.add_interaction("A", "B")
harmonic_bond.add_interaction("B", "B")

# create harmonic bias
harmonic_bias = HarmonicBias()
group = np.array([b.index for b in list(topology.chains)[0].beads])
harmonic_bias.add_interaction(group, k=0.5, r0=20.0, axis=0)

# create force field and system
force_field = ForceField()
force_field.add_potential(lj)
force_field.add_potential(harmonic_bond)
force_field.add_potential(harmonic_bias)
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator(step_size=0.002)

# create simulation and initialize
simulation = Simulation(system, integrator)
simulation.initialize()
simulation.minimize_energy()
simulation.system.topology.to_pdb(file_prefix + "initial.pdb",
                                  positions=simulation.state.positions)

# set up reporting
simulation.thermo_file = file_prefix+"_thermo.csv"
simulation.thermo_frequency = 10000
simulation.thermo_verbose = True
simulation.traj_file = file_prefix + "_traj.pdb"
simulation.traj_frequency = 10000
simulation.traj_min_image = True

# run
simulation.step(10000000)
