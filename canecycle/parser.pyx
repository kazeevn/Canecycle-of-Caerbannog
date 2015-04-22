from itertools import izip, imap
from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction
import string

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
    cdef HashFunction hash_function
    cdef list format

    def __cinit__(self, HashFunction hash_function, list format_):
        self.hash_function = hash_function
        self.format = format_
    
    cpdef Item parse(self, str line):
        cdef list processed_line
        cdef Item item
        cdef str readout
        cdef str column_name
        processed_line = map(string.strip, line.split(','))
        item = Item(self.hash_function.hash_size)
        column_namer = name_iterator()
        for item_format, readout, column_name in \
            izip(self.format, processed_line, column_namer):
            if readout == '':
                continue
            if item_format == ValueType_label:
                item.label = int(readout) * 2 - 1
            elif item_format == ValueType_categorical:
                item.features[self.hash_function.hash(
                    column_name + readout)] = 1
            elif item_format == ValueType_numerical:
                item.features[self.hash_function.hash(column_name)] = \
                    float(readout)
            elif item_format != ValueType_skip:
                raise ValueError("Invalid format %s" % item_format)
        return item
    
    cpdef unsigned int get_features_count(self):
        return self.hash_function.hash_size
                
        
