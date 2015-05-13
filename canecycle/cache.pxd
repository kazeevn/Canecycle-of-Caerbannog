cimport numpy as np

from canecycle.item cimport Item
from canecycle.source cimport Source

cdef class CacheWriter(object):
    cdef np.uint64_t objects_written
    cdef object item_array
    cdef object table
    cdef object file
    cdef np.uint64_t max_feature_columns_count
    cdef list struct_mapping
    cdef np.uint64_t hash_size

    cpdef open(self, filename)
    cpdef write_item(self, Item item)
    cpdef close(self)


cdef class CacheReader(Source):
    cdef object table
    cdef object file

    cpdef restart(self, np.int_t holdout)
    cpdef Item unpack_item(self, object row)
    cpdef close(self)
    cpdef np.uint64_t get_features_count(self)
