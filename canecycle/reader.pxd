from cpython cimport bool
from numpy cimport uint64_t

from canecycle.source cimport Source
from canecycle.parser cimport Parser
from canecycle.parser import read_shad_lsml_header
from canecycle.cache cimport CacheWriter

cdef class Reader(Source):
    cdef object file
    cdef Parser parser
    cdef str filename
    cdef uint64_t skip
    cdef str cache_file_name
    cdef CacheWriter cache_writer
    cdef bool open_cache_writer

    
    cpdef uint64_t get_features_count(self)
    cpdef uint64_t get_feature_columns_count(self)
    cpdef close(self)
