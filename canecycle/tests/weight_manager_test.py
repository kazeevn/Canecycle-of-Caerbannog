from __future__ import division

import unittest
import os.path

from canecycle.weight_manager import WeightManager 


class TestWeightManager(unittest.TestCase):
    def test_weight_increment_one(self):
        weight_manager = WeightManager() # p=0.5, n=1
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/3, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/5, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/7, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 7/3, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 7/5, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 1, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 7/9, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 9/11, places=5)
    
    def test_weight_increment_two(self):
        weight_manager = WeightManager(1, 0.5) # p=0.5, n=2
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/2, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/3, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1/4, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 2, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 4/3, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 1, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(1, 1), 4/5, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 1, places=5)
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 5/6, places=5)
    
    def test_weight_increment_three(self):
        weight_manager = WeightManager(100, 50) # p=0.5, n=200
        self.assertAlmostEqual(weight_manager.get_weight(-1, 1), 100/101, places=5)
