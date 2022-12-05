import numpy as np

from mdlib.bead_type import BeadType
from mdlib.topology import Bead, LinearChain, Topology
from mdlib.potentials import LJWCA, HarmonicBond
from mdlib.forcefield import ForceField
from mdlib.system import System
from mdlib.integrators import LangevinIntegrator
from mdlib.simulation import Simulation

# parameters
N = 240
N_p = 4
rho = 0.8
L = (N / rho)**(1/3)

# create topology
bead_type = BeadType("A")
homopolymer = LinearChain("lj_particle", N_p * [Bead("A", bead_type)])
topology = Topology()
topology.add_chain(homopolymer, n=int(N / N_p))
topology.box_lengths = np.array([L]*3)

# create lj and harmonic bond potentials
lj = LJWCA()
lj.add_interaction("A", "A", eps=1.0, sigma=1.0, cut=2.5, lambda_lj=1.0, lambda_wca=0.0)
harmonic_bond = HarmonicBond()
harmonic_bond.add_interaction("A", "A", k=3000.0, r0=1.0)

# create force field
force_field = ForceField()
force_field.add_potential(lj)
force_field.add_potential(harmonic_bond)


# create system
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator(step_size=0.002, temperature=1.0, friction_coefficient=1.0)

# create simulation and initialize
simulation = Simulation(system, integrator)
simulation.initialize()
simulation.minimize_energy()
simulation.system.topology.to_pdb("lj_homopolymer_initial.pdb", positions=simulation.state.positions)

# run
simulation.step(100000)
simulation.system.topology.to_pdb("lj_homopolymer_final.pdb", positions=simulation.state.positions)
