import unittest
import os.path
from itertools import imap

from canecycle.hash_function import HashFunction
from canecycle.parser import Parser, read_shad_lsml_header
from canecycle.reader import Reader, from_shad_lsml
from canecycle.source import NotInitialized

class TestReader(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")

    def test_throw(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(20)
        parser = Parser(hash_function, format_)
        reader = Reader(self.test_file, parser, skip=1)
        with self.assertRaises(NotInitialized):
            next(reader)
    

    def test_holdout_count(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(20)
        parser = Parser(hash_function, format_)
        reader = Reader(self.test_file, parser, skip=1)
        reader.restart(0)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 1000)
        reader.restart(2)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 500)
        reader.restart(-2)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 500)
        reader.restart(1)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 0)
        reader.restart(-1)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 1000)
        reader.restart(13)
        count_13 = sum(imap(lambda item: 1, reader))
        reader.restart(-13)
        count_13 += sum(imap(lambda item: 1, reader))
        self.assertEqual(count_13, 1000)

    def test_from_shad_lsml(self):
        reader = from_shad_lsml(self.test_file, 3)
        reader.restart(0)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 1000)
        self.assertEqual(reader.get_features_count(), 2**3)
