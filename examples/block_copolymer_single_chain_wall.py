import numpy as np

from projectlib import (
    Bead,
    BeadType,
    ForceField,
    HarmonicBond,
    LangevinIntegrator,
    LinearChain,
    LJWCA,
    Simulation,
    System,
    Topology,
    WallLJWCA
)

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
topology.periodicity[0] = False

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

# create wall potentials
wall_ljwca = WallLJWCA()
wall_ljwca.add_interaction("A", cut=7.5, lambda_lj=1.0, lambda_wca=0.0)
wall_ljwca.add_interaction("B", lambda_lj=0.0, lambda_wca=1.0)

# create force field and system
force_field = ForceField()
force_field.add_potential(lj)
force_field.add_potential(harmonic_bond)
force_field.add_potential(wall_ljwca)
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator()

# create simulation and initialize
simulation = Simulation(system, integrator)
simulation.initialize()
simulation.minimize_energy()
simulation.system.topology.to_pdb("block_copolymer_single_chain_wall_initial.pdb",
                                  positions=simulation.state.positions)
