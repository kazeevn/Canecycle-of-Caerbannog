import unittest
import numpy as np

from scipy.sparse import coo_matrix
from canecycle.optimizer import Optimizer
from canecycle.item import Item

class LossFunction(object):

    def get_gradient(self, point, label):
        return point * 2


class OptimizerTestCase(unittest.TestCase):
    def test_zero_vector(self):
        loss_function = LossFunction()
        item = Item()
        point = coo_matrix([0., 0., 0.])
        item.features = point
        optimizer = Optimizer(l1Regularization=0, l2Regularization=0,
                              stepSize=0.1, scaleDown=0.9, loss_function=loss_function)
        test_value = np.random.rand(3)
        point = optimizer.step(item, 1, test_value)
        np.testing.assert_array_almost_equal(point, test_value)

if __name__ == '__main__':
    unittest.main()
