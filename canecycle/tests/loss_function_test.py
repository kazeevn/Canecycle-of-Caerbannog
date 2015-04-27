import unittest
from canecycle.item import Item
from canecycle.loss_function import LossFunction
import numpy as np

class LossFunctionTest(unittest.TestCase):
    def test_predict_proba(self):
        item = Item()
        item.indexes = np.array([0, 1])
        item.data = np.array([1., 1.])
        weights = np.array([1., 1.])
        loss = LossFunction()
        self.assertAlmostEqual(loss.get_proba(item, weights), 1./(1. + np.exp(-2)))

    def test_predict_loss(self):
        item = Item()
        item.indexes = np.array([0, 1])
        item.data = np.array([1., 1.])
        weights = np.array([1., 1.])
        loss = LossFunction()
        self.assertAlmostEqual(loss.get_proba(item, weights), np.log(1./(1. + np.exp(-2)))) 



if __name__ == '__main__':
    unittest.main()

    