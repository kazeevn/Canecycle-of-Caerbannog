from itertools import izip, imap, islice, count
import numpy as np
cimport numpy as np

from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction


def name_iterator():
    cdef unsigned int i
    i = 0
    while True:
        yield "%d_" % i
        i += 1


VALUETYPE_LABEL = 0
VALUETYPE_CATEGORICAL = 1
VALUETYPE_NUMERICAL = 2
VALUETYPE_SKIP = 3


def transform_header(header_item):
    """Transformes header items from SHAD-LSML strings
    into internal enum"""
    if header_item == 'CLICK':
        return VALUETYPE_LABEL
    elif header_item == 'ID':
        return VALUETYPE_SKIP
    # It should be CAT, but the header contains typos
    elif header_item.startswith('CA') or header_item.startswith('AT'):
        return VALUETYPE_CATEGORICAL
    elif header_item.startswith('NUM'):
        return VALUETYPE_NUMERICAL
    else:
        raise ValueError("Unknown item in header %s" % header_item)


def read_shad_lsml_header(filename):
    """Reads header of SHAD-LSML files.
    Arguments:
    filename - file name to read
    Returnes: list of formats for Parser
    """
    
    input_file = open(filename)
    header = input_file.next().split(',')
    result =  map(transform_header, header)
    input_file.close()
    return result


cdef class Parser(object):
    """Class for parsing lines from SHAD-LSML into Item"""
    
    def __cinit__(self, HashFunction hash_function, list format_):
        """Arguments:
        hash_function - HashFunction to use
        format_ - list of ValueTypes to interpret lines
        """
        
        cdef str colomn_name
        self.hash_function = hash_function
        self.format = format_
        column_namer = name_iterator()
        self.column_names = list(islice(column_namer, len(self.format)))
        self.feature_columns_count = len(filter(
            lambda item_format: item_format in (VALUETYPE_CATEGORICAL, VALUETYPE_NUMERICAL),
            self.format))
        self.numeric_hashes = list()
        for column_name, item_format in izip(self.column_names, self.format):
            if item_format == VALUETYPE_NUMERICAL:
                self.numeric_hashes.append(self.hash_function.hash(column_name))
            else:
                # None would have been more appropriate, but Cython doesn't support
                self.numeric_hashes.append(0)
    

    cpdef Item parse(self, str line):
        """Parses a line into an Item.
        Arguments:
        line - line to parse
        Returnes: Item
        """
        
        cdef list processed_line
        cdef Item item
        cdef str readout
        cdef str column_name
        cdef list indexes
        cdef list data
        cdef np.uint64_t item_format
        cdef np.uint64_t hash
        cdef np.uint64_t index
        processed_line = line.rstrip().split(',')
        item = Item()
        item.indexes = np.ndarray(self.feature_columns_count, dtype=np.uint64)
        item.data = np.ndarray(self.feature_columns_count, dtype=np.float_)
        index = 0
        for item_format, readout, column_name, hash in \
            izip(self.format, processed_line, self.column_names, self.numeric_hashes):
            if readout == '':
                continue
            if item_format == VALUETYPE_LABEL:
                item.label = int(readout) * 2 - 1
            elif item_format == VALUETYPE_CATEGORICAL:
                item.indexes[index] = self.hash_function.hash(column_name + readout)
                item.data[index] = 1.
                index += 1
            elif item_format == VALUETYPE_NUMERICAL:
                # In __init__ we precalculate hashes for numeric values
                item.indexes[index] = hash
                item.data[index] = float(readout)
                index += 1
            elif item_format != VALUETYPE_SKIP:
                raise ValueError("Invalid format %s" % item_format)
        item.data.resize(index)
        item.indexes.resize(index)
        return item
    
    cpdef np.uint64_t get_features_count(self):
        """Returnes hash function values space size"""
        
        return self.hash_function.hash_size
