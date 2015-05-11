# cython: profile=True
from canecycle.source import NotInitialized, dropping_iterator
from canecycle.parser import (
    read_shad_lsml_header,
    ValueType_numerical,
    ValueType_skip)
from canecycle.hash_function cimport HashFunction
from canecycle.item cimport Item
from canecycle.item import Item
from itertools import imap
from canecycle.parser cimport Parser
cimport numpy as np
from canecycle.cache import CacheReader, CacheWriter

# Should be @staticmethod, but Cython doesn't support it in cpdef
# Should be cpdef, but Cython doesn't support closures
def from_shad_lsml(str filename, uint64_t hash_size, discard_numeric=False, cache_file_name=''):
    cdef list format
    cdef Parser parser
    cdef Reader reader
    format_ = read_shad_lsml_header(filename)
    if discard_numeric:
        format_ = map(
            lambda item_format: ValueType_skip if
            item_format == ValueType_numerical else item_format,
            format_)
    hash_function = HashFunction(hash_size)
    parser = Parser(hash_function, format_)
    reader = Reader(filename, parser, 1, cache_file_name)
    return reader


cdef class Reader(Source):
    def __iter__(self):
        if not self.is_ready:
            raise NotInitialized("You should call restart before "
                                 "using a source")
        return self.iterator


        
    def __cinit__(self, str filename, Parser parser, uint64_t skip, str cache_file_name=''):
        self.is_ready = False
        self.parser = parser
        self.skip = skip
        self.filename = filename
        self.cache_file_name = cache_file_name
        
    def restart(self, np.int_t holdout, bool write_cache=False, bool use_cache=False):
        """Restarts the source. Specify positive holdout to omit each h-th
        item negative to omit each but h-th item, zero to omit
        nothing. If called with write_cache, will write cache of the
        read lines - all, even held out. If called with use_cache,
        will read from the cache, not file.

        """
        if self.is_ready:
            self.file.close()
        else:
            self.is_ready = True
        self.file = open(self.filename)
        for _ in xrange(self.skip):
            next(self.file)
        self.holdout = holdout
        if write_cache and use_cache:
            raise ValueError("Can't write and use cache at the same time")
            
        if use_cache:
            cache_reader = CacheReader(self.cache_file_name)
            cache_reader.restart(holdout)
            self.iterator = cache_reader.__iter__()
        elif write_cache:
            cache_writer = CacheWriter(self.get_feature_columns_count(), self.get_features_count())
            cache_writer.open(self.cache_file_name)
            def parse_and_cache(str line):
                cdef Item item = self.parser.parse(line)
                cache_writer.write(item)
                return item
            self.iterator = dropping_iterator(imap(parse_and_cache, self.file), self.holdout)
        else:
            # About lambda. Cython refused to run without it.
            self.iterator = imap(lambda line: self.parser.parse(line),
                                 dropping_iterator(self.file, self.holdout))
            
    
    cpdef uint64_t get_features_count(self):
        return self.parser.get_features_count()

    
    cpdef close(self):
        self.file.close()


    cpdef uint64_t get_feature_columns_count(self):
        return self.parser.feature_columns_count
