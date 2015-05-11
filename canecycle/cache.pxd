cimport numpy as np

from canecycle.item cimport Item


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
