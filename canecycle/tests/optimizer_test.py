import unittest
import numpy as np

from scipy.sparse import coo_matrix
from canecycle.optimizer import Optimizer
from canecycle.item import Item
from canecycle.loss_function import LossFunction
   


class OptimizerTestCase(unittest.TestCase):
    def test_zero_vector(self):
        loss_function = LossFunction()
        item = Item()
        item.indexes = np.array([], dtype=np.uint64)
        item.data = np.array([])
        optimizer = Optimizer(l1Regularization=0, l2Regularization=0,
                              stepSize=0.1, scaleDown=0.9, loss_function=loss_function)
        test_value = np.random.rand(3)
        point = optimizer.step(item, 1, test_value)
        np.testing.assert_array_almost_equal(point, test_value)

    def test_minimization(self):
        loss_function = LossFunction()
        item = Item()
        item.indexes = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([1.0, 1.0], dtype=np.float_)
        weights = np.array([0., 0.])
        item.label = 1
        optimizer = Optimizer(l1Regularization=0, l2Regularization=0,
                              stepSize=0.1, scaleDown=0.9, loss_function=loss_function)
        loss = loss_function.get_loss(item, weights)
        weights = optimizer.step(item, 1, weights)
        new_loss = loss_function.get_loss(item, weights)
        self.assertLess(new_loss, loss)


if __name__ == '__main__':
    unittest.main()
