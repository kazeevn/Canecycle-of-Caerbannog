import os.path
from unittest import TestCase
from tempfile import NamedTemporaryFile
from itertools import izip, imap
import numpy as np

from canecycle.reader import from_shad_lsml
from canecycle.cache import CacheWriter, CacheReader


class TestReader(TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
   
    def test_cache_write_and_read(self):
        cache_file = './testing.cache'
        hash_size = 2**20
        reader = from_shad_lsml(self.test_file, hash_size)
        reader.restart(0)
        cache_writer = CacheWriter(60, hash_size)
        cache_writer.open(cache_file)
        for item in reader:
            cache_writer.write_item(item)
        cache_writer.close()
        
        reader.restart(3)
        cache_reader = CacheReader(cache_file)
        cache_reader.restart(3)
        self.assertEqual(hash_size, cache_reader.get_hash_size())
        for read_item, cached_item in izip(reader, cache_reader):
            self.assertEqual(read_item.label, cached_item.label)
            self.assertEqual(read_item.weight, cached_item.weight)
            np.testing.assert_array_equal(
                read_item.data, cached_item.data)
            np.testing.assert_array_equal(
                read_item.indexes, cached_item.indexes)

        reader.restart(-3)
        cache_reader.restart(-3)
        for read_item, cached_item in izip(reader, cache_reader):
            self.assertEqual(read_item.label, cached_item.label)
            self.assertEqual(read_item.weight, cached_item.weight)
            np.testing.assert_array_equal(
                read_item.data, cached_item.data)
            np.testing.assert_array_equal(
                read_item.indexes, cached_item.indexes)

        reader.close()
        cache_reader.restart(-4)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 250)
        cache_reader.restart(-2)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 500)
        cache_reader.restart(-100)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 10)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 0)

        cache_reader.restart(4)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 750)
        cache_reader.restart(2)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 500)
        cache_reader.restart(100)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 990)
        self.assertEqual(sum(imap(lambda item: 1, cache_reader)), 0)



        
