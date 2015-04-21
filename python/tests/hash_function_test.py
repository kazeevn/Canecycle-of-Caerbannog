import unittest
import hash_function


class TestHashFunction(unittest.TestCase):

    def test_uncut_points(self):
        hash_ = hash_function.HashFunction(63)
        self.assertEqual(hash_.hash("Kurchatov"), 6304225726192864752L)
        self.assertEqual(hash_.hash("Landau"), 3567067149801099079L)
        
