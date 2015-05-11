cimport numpy

from canecycle.item cimport Item

cdef class CacheWriter(object):
    cdef numpy.uint64_t objects_written
    cdef object item_array
    cdef object table
    cdef object file
    cdef numpy.uint64_t max_feature_columns_count
    cdef list struct_mapping
    cdef unsigned int hash_size

    cpdef open(self, filename)
    cpdef write_item(self, Item item)
    cpdef close(self)
