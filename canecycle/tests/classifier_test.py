import os.path
from unittest import TestCase

from canecycle.reader import from_shad_lsml


class TestClassifier(TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
    def test_fit(self):
        hash_size = 2**20
        reader = from_shad_lsml(self.test_file, hash_size)
