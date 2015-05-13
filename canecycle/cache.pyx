import tables
cimport numpy as np
import numpy as np
from itertools import imap

from canecycle.item cimport Item
from canecycle.source cimport Source
from canecycle.source import dropping_iterator


cdef class CacheWriter(object):
    """Class for writing .can binary files."""
    
    def __cinit__(self, max_feature_columns_count, hash_size):
        """Arguments:
        max_feature_columns_count - maximum number of features per item,
            used for static space allocation
        hash_size - size of hash function value space. Absolute value,
            not power of two
        """
        
        self.max_feature_columns_count = max_feature_columns_count
        self.struct_mapping = [
            ('label', 'int_'),
            ('weight', 'float_'),
            ('features_count', 'uint64'),
            ('indices', 'uint64', max_feature_columns_count),
            ('data', 'float', max_feature_columns_count)
        ]
        self.hash_size = hash_size

    cpdef open(self, filename):
        """Opens a file for writing.
        Arguments:
        filename - file name
        """
        
        cdef tuple table_description
        cdef np.ndarray mapped_item
        cdef np.ndarray metadata
        cdef object metadata_table
        cdef object filters

        filters = tables.Filters(complib='blosc', complevel=6)
        self.file = tables.open_file(filename, mode='w', chunkshape=(10000, 1), filters=filters)
        self.objects_written = 0
        mapped_item = np.ndarray(0, dtype=self.struct_mapping)
        metadata = np.ndarray((1, ), dtype=[('hash_size', 'uint64'),])
        metadata['hash_size'][0] = self.hash_size
        self.table = self.file.create_table(self.file.root, 'items_table',  mapped_item)
        metadata_table = self.file.create_table(self.file.root, 'metadata_table', metadata)
        metadata_table.close()

    cpdef write_item(self, Item item):
        """Writes an item to the .can file.
        Arguments:
        item - Item to write.
        """
        
        cdef np.uint64_t features_count
        mapped_item = np.ndarray(1, dtype=self.struct_mapping)
        mapped_item['label'] = item.label
        mapped_item['weight'] = item.weight
        features_count =  len(item.indices)
        mapped_item['features_count'] = features_count
        mapped_item['indices'][0, :features_count] = item.indices
        mapped_item['data'][0, :features_count] = item.data
        self.table.append(mapped_item)
        
    cpdef close(self):
        """Closes the .can file"""
        self.file.close()
    

cdef class CacheReader(Source):
    """The class to read from .can file as a Source"""
    
    def __cinit__(self, filename):
        """Arguments:
        filename - .can file to open
        """
        
        cdef object file_
        self.file = tables.open_file(filename)
        self.table = self.file.root.items_table
        self.is_ready = False

    cpdef restart(self, np.int_t holdout):
        """Restarts the source to read from the beginning.
        Agruments:
        holdout - if zero, does nothing, if positive, skip every h-th item,
            if negative, skip every but h-th item
        """
        
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
        """Converts row read from HDF5 into Item.
        Arguments:
        item - Item to convert
        """
        
        cdef np.uint64_t features_count
        item = Item()
        item.label = row['label']
        item.weight = row['weight']
        features_count = row['features_count']
        item.data = row['data']
        item.indices = row['indices']
        item.data.resize(features_count)
        item.indices.resize(features_count)
        return item
    
    cpdef close(self):
        """Closes the source"""

        self.file.close()

    cpdef np.uint64_t get_features_count(self):
        """Returnes feature space size (i. e. hash size)
        used while building the cache"""

        return self.file.root.metadata_table[0]['hash_size']

