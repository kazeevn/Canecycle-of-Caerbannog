from unittest import TestCase
from canecycle.source import dropping_iterator

class TestDroppingIterator(TestCase):
    test_list = [1, 2, 3, 4, 5, 6, 7, 8]

    def test_zero_holdout(self):
        dropped_list = list(dropping_iterator(self.test_list, 0))
        self.assertSequenceEqual(dropped_list, self.test_list)

    def test_positive_holdout(self):
        dropped_list = list(dropping_iterator(self.test_list, 3))
        self.assertSequenceEqual(dropped_list,
                                 [2, 3, 5, 6, 8])

    def test_negative_holdout(self):
        dropped_list = list(dropping_iterator(self.test_list, -3))
        self.assertSequenceEqual(dropped_list, [1, 4, 7])
