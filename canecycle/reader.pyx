from itertools import imap
cimport numpy as np
from cpython cimport bool

from canecycle.source import NotInitialized, dropping_iterator
from canecycle.parser import read_shad_lsml_header, VALUETYPE_NUMERICAL, VALUETYPE_SKIP
from canecycle.hash_function cimport HashFunction
from canecycle.item cimport Item
from canecycle.item import Item
from canecycle.parser cimport Parser
from canecycle.cache import CacheReader, CacheWriter


# Should be @staticmethod, but Cython doesn't support it in cpdef
# Should be cpdef, but Cython doesn't support closures
def from_shad_lsml(str filename, np.uint64_t hash_size,
                   bool discard_numeric=False, str cache_file_name=''):
    """Convenient function to open files in SHAD-LSML format.
    Arguments:
    filename - file name to open
    hash_size - hash size to use, absolute value, not power of 2
    discard_numeric - discard numeric features
    cache_file_name - instruct Reader to use cache file with the name
    Returns Reader instance.
    """
    
    cdef list format
    cdef Parser parser
    cdef Reader reader
    format_ = read_shad_lsml_header(filename)
    if discard_numeric:
        format_ = map(
            lambda item_format: VALUETYPE_SKIP if
            item_format == VALUETYPE_NUMERICAL else item_format,
            format_)
    hash_function = HashFunction(hash_size)
    parser = Parser(hash_function, format_)
    reader = Reader(filename, parser, 1, cache_file_name)
    return reader


cdef class Reader(Source):
    """Class for reading SHAD-LSML files. Supports caching in .can,
    improves overall speed by ~30%. Don't forget to call restart before
    using.
    """
    def __iter__(self):
        if not self.is_ready:
            raise NotInitialized("You should call restart before "
                                 "using a source")
        return self.iterator

    def __cinit__(self, str filename, Parser parser,
                  np.uint64_t skip, str cache_file_name=''):
        """Arguments:
        filename - file name to open
        parser - Parser instance to use
        skip - skip line in the beginning of the file
        cache_file_name - use cache file at location
        """

        self.is_ready = False
        self.parser = parser
        self.skip = skip
        self.filename = filename
        self.cache_file_name = cache_file_name
        self.open_cache_writer = False
        self.open_cache_reader = False
        
    def restart(self, np.int_t holdout, bool write_cache=False, bool use_cache=False):
        """Restarts the source to read from the beginning.
        Arguments:
        holdout - to omit each h-th item negative to omit each but h-th item,
        zero to omit nothing.
        write_cache - will write cache of the read lines - all, even held out.
        use_cache - will read from the cache, not file.
        """

        if self.is_ready:
            self.file.close()
        else:
            self.is_ready = True
        
        if self.open_cache_writer:
            self.open_cache_writer = False
            self.cache_writer.close()
        
        if self.open_cache_reader:
            self.open_cache_reader = False
            self.cache_reader.close()

        self.file = open(self.filename)
        for _ in xrange(self.skip):
            next(self.file)

        self.holdout = holdout
        if write_cache and use_cache:
            raise ValueError("Can't write and use cache at the same time")
            
        if use_cache:
            self.cache_reader = CacheReader(self.cache_file_name)
            self.cache_reader.restart(holdout)
            self.open_cache_reader = True
            self.iterator = self.cache_reader.__iter__()
        elif write_cache:
            self.open_cache_writer = True
            self.cache_writer = CacheWriter(self.get_feature_columns_count(),
                                            self.get_features_count())
            self.cache_writer.open(self.cache_file_name)
            def parse_and_cache(str line):
                cdef Item item = self.parser.parse(line)
                self.cache_writer.write_item(item)
                return item
            self.iterator = dropping_iterator(imap(parse_and_cache, self.file), self.holdout)
        else:
            self.iterator = imap(lambda line: self.parser.parse(line),
                                 dropping_iterator(self.file, self.holdout))

    cpdef np.uint64_t get_features_count(self):
        """Returns the hash function value space size"""

        return self.parser.get_features_count()
    
    cpdef close(self):
        """Closes the source"""

        if self.open_cache_writer:
            self.cache_writer.close()
        if self.open_cache_reader:
            self.cache_reader.close()
        self.file.close()

    cpdef np.uint64_t get_feature_columns_count(self):
        """Returns maximum number of features per item"""

        return self.parser.feature_columns_count
