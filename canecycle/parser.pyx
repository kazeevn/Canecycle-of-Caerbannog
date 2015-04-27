# cython: profile=True
from itertools import izip, imap
from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction
from itertools import islice, count
import string
cimport numpy
import numpy

def name_iterator():
    cdef unsigned int i
    i = 0
    while True:
        yield "%d_" % i
        i += 1

#TODO(kazeevn) switch to proper Cython
ValueType_label = 0
ValueType_categorical = 1
ValueType_numerical = 2
ValueType_skip = 3

def transform_header(header_item):
    if header_item == 'CLICK':
        return ValueType_label
    elif header_item == 'ID':
        return ValueType_skip
    # It should be CAT, but the header contains typos
    elif header_item.startswith('CA') or header_item.startswith('AT'):
        return ValueType_categorical
    elif header_item.startswith('NUM'):
        return ValueType_numerical
    else:
        raise ValueError("Unknown item in header %s" % header_item)


def read_shad_lsml_header(filename):
    input_file = open(filename)
    header = input_file.next().split(',')
    result =  map(transform_header, header)
    input_file.close()
    return result


cdef class Parser:
    def __cinit__(self, HashFunction hash_function, list format_):
        cdef str colomn_name
        self.hash_function = hash_function
        self.format = format_
        column_namer = name_iterator()
        self.column_names = list(islice(column_namer, len(self.format)))
        self.feature_columns_count = len(filter(
            lambda item_format: item_format in (ValueType_categorical, ValueType_numerical),
            self.format))
        self.numeric_hashes = list()
        for column_name, item_format in izip(self.column_names, self.format):
            if item_format == ValueType_numerical:
                self.numeric_hashes.append(self.hash_function.hash(column_name))
            else:
                # None would have been more appropriate, but Cython doesn't support
                self.numeric_hashes.append(0)
    
    cpdef Item parse(self, str line):
        cdef list processed_line
        cdef Item item
        cdef str readout
        cdef str column_name
        cdef list indexes
        cdef list data
        cdef unsigned int item_format
        cdef unsigned long hash
        cdef unsigned long index
        processed_line = line.rstrip().split(',')
        item = Item()
        item.indexes = numpy.ndarray(self.feature_columns_count, dtype=numpy.uint64)
        item.data = numpy.ndarray(self.feature_columns_count, dtype=numpy.float_)
        index = 0
        for item_format, readout, column_name, hash in \
            izip(self.format, processed_line, self.column_names, self.numeric_hashes):
            if readout == '':
                if item_format in (ValueType_categorical, ValueType_numerical):
                    index += 1
                continue
            if item_format == ValueType_label:
                item.label = int(readout) * 2 - 1
            elif item_format == ValueType_categorical:
                item.indexes[index] = self.hash_function.hash(column_name + readout)
                item.data[index] = 1.
                index += 1
            elif item_format == ValueType_numerical:
                # In __init__ we precalculate hashes for numeric values
                item.indexes[index] = hash
                item.data[index] = float(readout)
                index += 1
            elif item_format != ValueType_skip:
                raise ValueError("Invalid format %s" % item_format)
        return item
    
    cpdef unsigned int get_features_count(self):
        return self.hash_function.hash_size
                
        
