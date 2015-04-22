import unittest
import numpy
import os.path
from itertools import imap

from canecycle.hash_function import HashFunction
from canecycle.reader import Reader, read_shad_lsml_header
import canecycle.reader

class TestReader(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
    
    def test_header_readout(self):
        header = read_shad_lsml_header(self.test_file)
        self.assertEqual(len(header), 1 + 37 + 23)
        self.assertEqual(header[0], canecycle.reader.ValueType_label)
        for header_type in header[1:37]:
            self.assertEqual(header_type, canecycle.reader.ValueType_categorical)
        for header_type in header[38:]:
            self.assertEqual(header_type, canecycle.reader.ValueType_numerical)

    def test_restart(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(20)
        reader = Reader(hash_function, self.test_file, format_, 1)
        items_count_no_restart = sum(map(lambda item: 1, reader))
        self.assertEqual(items_count_no_restart, 1000)
        reader.restart()
        items_count_restart = sum(map(lambda item: 1, reader))
        self.assertEqual(items_count_restart, 1000)
