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
        self.assertEqual(header[0], canecycle.parser.ValueType_label)
        for header_type in header[1:37]:
            self.assertEqual(header_type, canecycle.parser.ValueType_categorical)
        for header_type in header[38:]:
            self.assertEqual(header_type, canecycle.parser.ValueType_numerical)

    def test_restart(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(20)
        parser = Parser(hash_function, format_)
        input_file = open(self.test_file)
        # Skipt header
        input_file.next()
        for line in input_file:
            item = parser.parse(line)
