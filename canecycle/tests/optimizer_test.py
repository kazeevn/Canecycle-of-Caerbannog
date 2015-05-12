import unittest
import numpy as np

from canecycle.optimizer import Optimizer
from canecycle.item import Item
from canecycle.loss_function import LossFunction


class OptimizerTestCase(unittest.TestCase):
    def test_zero_vector(self):
        loss_function = LossFunction()
        item = Item()
        item.indices = np.array([], dtype=np.uint64)
        item.data = np.array([])
        optimizer = Optimizer(0, 0, 3, 0.1, 0.1, loss_function)
        test_value = np.random.rand(3)
        point = optimizer.step(item, test_value)
        np.testing.assert_array_almost_equal(point, test_value)

    def test_minimization(self):
        loss_function = LossFunction()
        item = Item()
        item.indices = np.array([0, 1], dtype=np.uint64)
        item.data = np.array([1.0, 1.0], dtype=np.float_)
        weights = np.array([0., 0.])
        item.label = 1
        optimizer = Optimizer(0, 0, 3, 0.1, 0.1, loss_function)
        old_loss = loss_function.get_loss(item, weights)
        weights = optimizer.step(item, weights)
        for counter in range(100):
            weights = optimizer.step(item, weights)
            new_loss = loss_function.get_loss(item, weights)
            self.assertLess(new_loss, old_loss)
            old_loss = new_loss


if __name__ == '__main__':
    unittest.main()
