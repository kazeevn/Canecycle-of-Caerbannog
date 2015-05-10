import unittest
from canecycle.item import Item
from canecycle.loss_function import LossFunction
import numpy as np

class LossFunctionTest(unittest.TestCase):
    def test_predict_proba(self):
        item = Item()
        item.indexes = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([1., 1.])
        weights = np.array([1., 1.])
        loss = LossFunction()
        self.assertAlmostEqual(
            loss.get_proba(item, weights), 1./(1. + np.exp(-2)))
        self.assertAlmostEqual(
            loss.get_log_proba(item, weights), np.log(1./(1. + np.exp(-2))))
        self.assertAlmostEqual(
            loss.get_log_one_minus_proba(item, weights), np.log(1 - 1./(1. + np.exp(-2))))



    def test_predict_loss(self):
        item = Item()
        item.indexes = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([1., 1.])
        weights = np.array([1., 1.])
        item.label = 1
        loss = LossFunction()
        self.assertAlmostEqual(
            loss.get_loss(item, weights), -np.log(1./(1. + np.exp(-2)))) 

    def test_proba_underflow(self):
        item = Item()
        item.indexes = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([-1e10, -1e10])
        weights = np.array([1., 1.])
        item.label = 1
        loss = LossFunction()
        self.assertAlmostEqual(
            loss.get_loss(item, weights), 2e10) 
        self.assertAlmostEqual(
            loss.get_log_proba(item, weights), -2e10) 
        self.assertAlmostEqual(
            loss.get_proba(item, weights), 0.0) 

    def test_proba_overflow(self):
        item = Item()
        item.indexes = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([1e6, 1e6])
        weights = np.array([1., 1.])
        item.label = 0
        loss = LossFunction()
        self.assertAlmostEqual(
            loss.get_loss(item, weights), 2e6) 
        self.assertAlmostEqual(
            loss.get_log_one_minus_proba(item, weights), -2e6) 
        self.assertAlmostEqual(
            loss.get_proba(item, weights), 1.0)
        item.label = 1
        self.assertEqual(
            loss.get_loss(item, weights), 0) 

if __name__ == '__main__':
    unittest.main()

    
