import unittest

from projectlib.bead_type import BeadType


class TestBeadType(unittest.TestCase):

    def test_n_bead_types(self):
        a = BeadType("A")
        self.assertEqual(1, BeadType.n_bead_types)
        b = BeadType("B")
        self.assertEqual(2, BeadType.n_bead_types)


if __name__ == '__main__':
    unittest.main()
