import numpy as np

from mdlib.bead_type import BeadType
from mdlib.topology import Bead, Chain, Topology
from mdlib.potentials import WallLJWCA, LJWCA
from mdlib.forcefield import ForceField
from mdlib.system import System
from mdlib.integrators import LangevinIntegrator
from mdlib.simulation import Simulation

# parameters
N = 240
rho = 0.8
L = (N / rho)**(1/3)

# create topology
bead_type = BeadType("A")
lj_bead = Chain("lj_particle", [Bead("A", bead_type)])
topology = Topology()
topology.add_chain(lj_bead, N)
topology.box_lengths = np.array([L]*3)
topology.periodicity[0] = False

# create lj potential and wall potentials
lj = LJWCA()
lj.add_interaction("A", "A", eps=1.0, sigma=1.0, cut=2.5, lambda_lj=1.0, lambda_wca=0.0)
wall_lj = WallLJWCA()
wall_lj.add_interaction("A", eps=1.0, sigma=0.5, cut=3.75, lower_bound=0.0, upper_bound=L)

# create force field
force_field = ForceField()
force_field.add_potential(lj)
force_field.add_potential(wall_lj)

# create system
system = System(topology, force_field)

# create integrator
integrator = LangevinIntegrator(step_size=0.002, temperature=1.0, friction_coefficient=1.0)

# create simulation and initialize
simulation = Simulation(system, integrator)
simulation.initialize()
simulation.minimize_energy()
simulation.system.topology.to_pdb("lj_fluid_wall_initial.pdb", positions=simulation.state.positions)

# set up reporting
simulation.thermo_file = "lj_fluid_wall_thermo.csv"
simulation.thermo_frequency = 1000
simulation.thermo_verbose = True
simulation.traj_file = "lj_fluid_wall_traj.pdb"
simulation.traj_frequency = 1000
simulation.traj_min_image = True

# run
simulation.step(10000)
