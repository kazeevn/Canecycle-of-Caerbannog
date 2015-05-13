from cpython cimport bool
cimport numpy as np

from canecycle.source cimport Source
from canecycle.parser cimport Parser
from canecycle.cache cimport CacheWriter, CacheReader


cdef class Reader(Source):
    cdef object file
    cdef Parser parser
    cdef str filename
    cdef np.uint64_t skip
    cdef str cache_file_name
    cdef CacheWriter cache_writer
    cdef CacheReader cache_reader
    cdef bool open_cache_writer
    cdef bool open_cache_reader
    
    cpdef np.uint64_t get_features_count(self)
    cpdef np.uint64_t get_feature_columns_count(self)
    cpdef close(self)
