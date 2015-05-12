import unittest
import numpy as np

from canecycle.hash_function import HashFunction


class TestHashFunction(unittest.TestCase):
    def test_uncut_points(self):
        hash_ = HashFunction(2**63)
        self.assertEqual(hash_.hash("Kurchatov"), 6304225726192864752L)
        self.assertEqual(hash_.hash("Landau"), 3567067149801099079L)

    def test_range(self):
        np.random.seed(42)
        for hash_size_bits in xrange(10):
            hash_ = HashFunction(2**hash_size_bits)
            for value in map(str, np.random.randint(1775, size=100)):
                self.assertLess(hash_.hash(value), 2**hash_size_bits)
