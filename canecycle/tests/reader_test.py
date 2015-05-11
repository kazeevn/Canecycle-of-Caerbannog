import unittest
import os.path
from itertools import imap, izip
import numpy

from canecycle.hash_function import HashFunction
from canecycle.parser import Parser, read_shad_lsml_header
from canecycle.reader import Reader, from_shad_lsml
from canecycle.source import NotInitialized
from canecycle.cache import CacheReader


class TestReader(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
    test_cache_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'test_train_except.can')
    def test_throw(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(2**20)
        parser = Parser(hash_function, format_)
        reader = Reader(self.test_file, parser, skip=1)
        with self.assertRaises(NotInitialized):
            next(reader.__iter__())
    
    def compare_items(self, item_one, item_two):
        self.assertEqual(item_one.label, item_two.label)
        self.assertEqual(item_one.weight, item_two.weight)
        numpy.testing.assert_array_equal(
            item_one.data, item_two.data)
        numpy.testing.assert_array_equal(
            item_one.indexes, item_two.indexes)

            
    def test_holdout_count(self):
        format_ = read_shad_lsml_header(self.test_file)
        hash_function = HashFunction(2**20)
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
        reader.restart(4)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 750)
        reader.restart(-1)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 1000)
        reader.restart(13)
        count_13 = sum(imap(lambda item: 1, reader))
        reader.restart(-13)
        count_13 += sum(imap(lambda item: 1, reader))
        self.assertEqual(count_13, 1000)

    def test_from_shad_lsml(self):
        reader = from_shad_lsml(self.test_file, 2**3)
        reader.restart(0)
        self.assertEqual(sum(imap(lambda item: 1, reader)), 1000)
        self.assertEqual(reader.get_features_count(), 2**3)

    def test_numeric_discard(self):
        reader = from_shad_lsml(self.test_file, 2**25, True)
        reader.restart(0)
        self.assertEqual(reader.get_feature_columns_count(), 37)

    def test_cache_writing(self):
        reader = from_shad_lsml(self.test_file, 2**25, True, self.test_cache_file)
        raw_reader = from_shad_lsml(self.test_file, 2**25, True)
        raw_reader.restart(3)
        reader.restart(3, write_cache=True)
        for read_item, cached_item in izip(raw_reader, reader):
            self.compare_items(read_item, cached_item)
        reader.close()

        cache_reader = CacheReader(self.test_cache_file)
        cache_reader.restart(0)
        raw_reader.restart(0)
        for read_item, cached_item in izip(raw_reader, cache_reader):
            self.compare_items(read_item, cached_item)
            
        reader.restart(-4, use_cache=True)
        raw_reader.restart(-4)
        for read_item, cached_item in izip(raw_reader, reader):
            self.compare_items(read_item, cached_item)
        
        
