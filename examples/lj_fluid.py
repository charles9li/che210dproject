import numpy as np

from projectlib.bead_type import BeadType
from projectlib.topology import Bead, Chain, Topology
from projectlib.potentials import LJWCA
from projectlib.forcefield import ForceField
from projectlib.system import System
from projectlib.integrators import LangevinIntegrator
from projectlib.simulation import Simulation

# parameters
N = 240
rho = 0.8
L = (N / rho)**(1/3)

# create topology
print("creating topology")
bead_type = BeadType("A")
lj_bead = Chain("lj_particle", [Bead("A", bead_type)])
topology = Topology()
topology.add_chain(lj_bead, N)
topology.box_lengths = np.array([L]*3)

# create lj potential
lj = LJWCA()
lj.add_interaction("A", "A", eps=1.0, sigma=1.0, cut=2.5, lambda_lj=1.0, lambda_wca=0.0)

# create force field
force_field = ForceField()
force_field.add_potential(lj)

# create system
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator(step_size=0.002, temperature=1.0, friction_coefficient=1.0)

# create simulation and initialize
simulation = Simulation(system, integrator)
simulation.initialize()
simulation.minimize_energy()
simulation.system.topology.to_pdb("lj_fluid_initial.pdb", positions=simulation.state.positions)

# # run
simulation.step(100000)
simulation.system.topology.to_pdb("lj_fluid_final.pdb", positions=simulation.state.positions)
