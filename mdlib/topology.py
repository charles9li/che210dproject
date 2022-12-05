"""topology.py: Used for storing topological information about a system."""
__all__ = ['Bead', 'Chain', 'LinearChain', 'Topology']

from copy import deepcopy

import numpy as np

from mdlib.bead_type import BeadType


class Bead(object):

    def __init__(self, name, bead_type):
        self.name = name
        if isinstance(bead_type, str):
            self.bead_type = BeadType.get_by_name(bead_type)
        else:
            self.bead_type = bead_type
        self.index_in_chain = None
        self.index = None


class Chain(object):

    def __init__(self, name, beads):
        self.name = name
        self._beads = []
        for i, bead in enumerate(beads):
            bead = deepcopy(bead)
            bead.index_in_chain = i
            self._beads.append(bead)
        self._bonds = []
        self.index = None
        self.packmol_instructions = []

    def add_bond(self, bead1, bead2):
        self._bonds.append((bead1, bead2))

    def _default_positions(self):
        _positions = np.zeros((self.n_beads, 3), dtype=float)
        _positions[:, 0] = np.arange(self.n_beads)
        return _positions * 0.5

    def to_pdb(self, filename):
        # create temporary topology
        t = Topology()

        # add this chain to the topology
        t.add_chain(self, n=1)

        # output to pdb
        t.to_pdb(filename, model_index=None, positions=None, append=False)

    def _pdb_str(self, positions=None, start_bead_index=1):
        if positions is None:
            positions = self._default_positions()
        positions *= 10.0
        s = ""
        bead_index = start_bead_index
        chain_name = self.name[:3].upper()
        chain_id = chr(ord("A") + self.index % 26)
        for bead in self.beads:
            bead_position = positions[bead.index_in_chain]
            if len(bead.name) < 4 and bead.name[:1].isalpha():
                bead_name = ' '+bead.name
            else:
                bead_name = bead.name[:4]
            s += f"HETATM{bead_index:>5d} {bead_name:<4s} {chain_name:>3s} {chain_id}{self.index+1:>4d}    " \
                 f"{bead_position[0]:>8.3f}{bead_position[1]:>8.3f}{bead_position[2]:>8.3f}  " \
                 f"1.00  0.00          EP\n"
            bead_index += 1
        s += f"TER   {bead_index:>5d}      {chain_name:>3s} {chain_id}{self.index+1:>4d}\n"
        return s

    @property
    def beads(self):
        return iter(self._beads)

    @property
    def bonds(self):
        return iter(self._bonds)

    @property
    def n_beads(self):
        return len(self._beads)

    @property
    def n_bonds(self):
        return len(self._bonds)


class LinearChain(Chain):

    def __init__(self, name, beads):
        super(LinearChain, self).__init__(name, beads)
        for bead1, bead2 in zip(self._beads[:-1], self._beads[1:]):
            self.add_bond(bead1, bead2)

    def _default_positions(self):
        _positions = np.zeros((self.n_beads, 3), dtype=float)
        for i in range(1, self.n_beads):
            _positions[i, :] = _positions[i-1, :]
            _positions[i, i % 3] += 1.0
        return _positions * 0.5


class Topology(object):

    def __init__(self):
        self._chain_types = {}
        self._chains = []
        self._n_chains = 0
        self._n_beads = 0
        self._bonds = []
        self._box_lengths = None
        self._periodicity = np.ones(3, dtype=bool)

    def add_chain(self, chain, n=1):
        self._chain_types[chain] = n

        for _ in range(n):
            # copy chain
            chain_copy = deepcopy(chain)

            # update indices of each bead
            for bead in chain_copy.beads:
                bead.index = bead.index_in_chain + self._n_beads

            # update index of chain
            chain_copy.index = self._n_chains

            # add chain and bonds to topology
            self._chains.append(chain_copy)
            for bond in chain_copy.bonds:
                self._bonds.append(bond)

            # update chain and bead counts
            self._n_chains += 1
            self._n_beads += chain_copy.n_beads

    @property
    def chain_types(self):
        return self._chain_types.keys()

    @property
    def chain_nums(self):
        return self._chain_types.values()

    @property
    def chains(self):
        return iter(self._chains)

    @property
    def beads(self):
        for chain in self.chains:
            for bead in chain.beads:
                yield bead

    @property
    def bonds(self):
        return iter(self._bonds)

    @property
    def n_chains(self):
        return self._n_chains

    @property
    def n_beads(self):
        return self._n_beads

    @property
    def n_bonds(self):
        return len(self._bonds)

    @property
    def box_lengths(self):
        return self._box_lengths

    @box_lengths.setter
    def box_lengths(self, new_box_lengths):
        self._box_lengths = new_box_lengths

    @property
    def periodicity(self):
        return self._periodicity

    def _pdb_str(self, model_index=None, positions=None, min_image=True):
        s = ""
        if self.box_lengths is not None:
            a = self.box_lengths[0] * 10.0
            b = self.box_lengths[1] * 10.0
            c = self.box_lengths[2] * 10.0
            s += f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}  90.00  90.00  90.00 P 1           1\n"
        if model_index is not None:
            s += f"MODEL     {model_index:>4d}\n"
        start_bead_index = 1
        for chain in self.chains:
            chain_positions = None
            if positions is not None:
                indices = [b.index for b in chain.beads]
                chain_positions = positions[indices, :]
            if self.box_lengths is not None and min_image:
                center_position = np.mean(chain_positions, axis=0)
                chain_positions -= self.box_lengths * np.floor(center_position / self.box_lengths)
            s += chain._pdb_str(positions=chain_positions, start_bead_index=start_bead_index)
            start_bead_index += chain.n_beads + 1
        if model_index is None:
            s += "END\n"
        else:
            s += "ENDMDL\n"
        return s

    def to_pdb(self, filename, model_index=None, positions=None, append=False, min_image=True):
        s = self._pdb_str(model_index=model_index, positions=positions, min_image=min_image)
        if append:
            mode = 'a'
        else:
            mode = 'w'
        with open(filename, mode) as f:
            f.write(s)
