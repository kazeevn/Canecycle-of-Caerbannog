import os.path
from unittest import TestCase
from tempfile import NamedTemporaryFile
from itertools import izip
import numpy

from canecycle.reader import from_shad_lsml
from canecycle.cache import CacheWriter, CacheReader

class TestReader(TestCase):
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
   
    def test_cache_write_and_read(self):
        cache_file = './testing.cache'
        hash_size = 20
        reader = from_shad_lsml(self.test_file, hash_size)
        reader.restart(0)
        cache_writer = CacheWriter(60, hash_size)
        cache_writer.open(cache_file)
        for item in reader:
            cache_writer.write_item(item)
        cache_writer.close()
        
        reader.restart(0)
        cache_reader = CacheReader(cache_file)
        self.assertEqual(hash_size, cache_reader.get_hash_size())
        for read_item, cached_item in izip(reader, cache_reader):
            self.assertEqual(read_item.label, cached_item.label)
            self.assertEqual(read_item.weight, cached_item.weight)
            numpy.testing.assert_array_equal(read_item.data, cached_item.data)
            numpy.testing.assert_array_equal(read_item.indexes, cached_item.indexes)
        cache_reader.close()
