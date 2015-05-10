import tables
cimport numpy
import numpy
from canecycle.item cimport Item
from canecycle.source cimport Source
from itertools import imap
from cpython cimport bool
from canecycle.source import dropping_iterator


cdef class CacheWriter(object):
    cdef numpy.uint64_t objects_written
    cdef object item_array
    cdef object table
    cdef object file
    cdef numpy.uint64_t max_feature_columns_count
    cdef list struct_mapping
    cdef unsigned int hash_size

    def __cinit__(self, max_feature_columns_count, hash_size):
        self.max_feature_columns_count = max_feature_columns_count
        # We serialize into
        # int_t Label, float_t weight, uint64_t[max_feature_columns_count] indexes,
        # float_t[max_feature_columns_count] data
        self.struct_mapping = [
            ('label', 'int_'),
            ('weight', 'float_'),
            ('features_count', 'uint64'),
            ('indexes', 'uint64', max_feature_columns_count),
            ('data', 'float', max_feature_columns_count)
        ]
        self.hash_size = hash_size

    cpdef open(self, filename):
        cdef tuple table_description
        cdef numpy.ndarray mapped_item
        cdef numpy.ndarray metadata
        cdef object metadata_table
        cdef object filters

        filters = tables.Filters(complib='blosc5')
        self.file = tables.open_file(filename, mode='w', chunkshape=(10000, 1), filters=filters)
        self.objects_written = 0
        mapped_item = numpy.ndarray(0, dtype=self.struct_mapping)
        metadata = numpy.ndarray((1, ), dtype=[('hash_size', 'uint64'),])
        metadata['hash_size'][0] = self.hash_size
        self.table = self.file.create_table(self.file.root, 'items_table',  mapped_item)
        metadata_table = self.file.create_table(self.file.root, 'metadata_table', metadata)
        metadata_table.close()

    cpdef write_item(self, Item item):
        cdef numpy.uint64_t features_count
        mapped_item = numpy.ndarray(1, dtype=self.struct_mapping)
        mapped_item['label'] = item.label
        mapped_item['weight'] = item.weight
        features_count =  len(item.indexes)
        mapped_item['features_count'] = features_count
        mapped_item['indexes'][0, :features_count] = item.indexes
        mapped_item['data'][0, :features_count] = item.data
        self.table.append(mapped_item)
        
    cpdef close(self):
        self.file.close()

    

cdef class CacheReader(Source):
    cdef object table
    cdef object file
    
    def __cinit__(self, filename):
        cdef object file_
        self.file = tables.open_file(filename)
        self.table = self.file.root.items_table
        self.is_ready = False

    cpdef restart(self, numpy.int64_t holdout):
        self.holdout = holdout
        if holdout > 0:
            self.iterator = imap(
                self.unpack_item,
                dropping_iterator(self.table.iterrows(), holdout)
            )
        elif holdout < 0:
            self.iterator = imap(
                self.unpack_item, self.table.iterrows(step=-holdout)
            )
        else:
            self.iterator = imap(
                self.unpack_item, self.table.iterrows())

        self.is_ready = True

    
    cpdef Item unpack_item(self, object row):
        cdef numpy.uint64_t features_count
        item = Item()
        item.label = row['label']
        item.weight = row['weight']
        features_count = row['features_count']
        item.data = row['data']
        item.indexes = row['indexes']
        item.data.resize(features_count)
        item.indexes.resize(features_count)
        return item
    
    cpdef close(self):
        self.file.close()

    cpdef numpy.uint64_t get_hash_size(self):
        cdef object metadata_table = \
            self.file.root.metadata_table
        return metadata_table[0]['hash_size']

    cpdef numpy.uint64_t get_features_count(self):
        return 2**self.get_hash_size()
