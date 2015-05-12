import unittest
import numpy
import os.path
from itertools import imap

from canecycle.hash_function import HashFunction
from canecycle.parser import Parser, read_shad_lsml_header
import canecycle.parser


class TestParser(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
    
    def test_header_readout(self):
        header = read_shad_lsml_header(self.test_file)
        self.assertEqual(len(header), 1 + 37 + 23)
        self.assertEqual(header[0], canecycle.parser.VALUETYPE_LABEL)
        for header_type in header[1:37]:
            self.assertEqual(header_type, canecycle.parser.VALUETYPE_CATEGORICAL)
        for header_type in header[38:]:
            self.assertEqual(header_type, canecycle.parser.VALUETYPE_NUMERICAL)

    def test_parse(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(2**20)
        parser = Parser(hash_function, format_)
        input_file = open(self.test_file)
        # Skip header
        input_file.next()
        line = input_file.next()
        print line
        item = parser.parse(line)
        self.assertEqual(item.label, -1)
        self.assertEqual(item.weight, 1)
        self.assertEqual(item.indexes[-1], 107535)
        self.assertEqual(item.data[-1], 1382)
        self.assertEqual(item.indexes[0], 641047)
        self.assertEqual(item.data[0], 1)
        self.assertEqual(item.data.sum(), 4437)
        for line in input_file:
            item = parser.parse(line)

    def test_features_count(self):
        hash_function = HashFunction(2**20)
        parser = Parser(hash_function, [])
        self.assertEqual(parser.get_features_count(), 2**20)
