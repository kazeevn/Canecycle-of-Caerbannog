import unittest
import hash_function
import numpy

class TestHashFunction(unittest.TestCase):

    def test_uncut_points(self):
        hash_ = hash_function.HashFunction(63)
        self.assertEqual(hash_.hash("Kurchatov"), 6304225726192864752L)
        self.assertEqual(hash_.hash("Landau"), 3567067149801099079L)
        

    def test_range(self):
        numpy.random.seed(42)
        for hash_size in xrange(10):
            hash_ = hash_function.HashFunction(hash_size)
            for value in map(str, numpy.random.randint(1775, size=100)):
                self.assetLess(hash_.hash(value), 2**hash_size)
        
