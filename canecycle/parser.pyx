from itertools import izip, imap
from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction
from itertools import islice
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
    def __cinit__(self, HashFunction hash_function, list format_):
        self.hash_function = hash_function
        self.format = format_
        column_namer = name_iterator()
        self.column_names = list(islice(column_namer, len(self.format)))
    
    cpdef Item parse(self, str line):
        cdef list processed_line
        cdef Item item
        cdef str readout
        cdef str column_name
        cdef dict features
        cdef unsigned int item_format
        processed_line = line.rstrip().split(',')
        item = Item(self.hash_function.hash_size)
        features = dict()
        for item_format, readout, column_name in \
            izip(self.format, processed_line, self.column_names):
            if readout == '':
                continue
            if item_format == ValueType_label:
                item.label = int(readout) * 2 - 1
            elif item_format == ValueType_categorical:
                features[self.hash_function.hash(
                    column_name + readout), 0] = 1.
            elif item_format == ValueType_numerical:
                features[self.hash_function.hash(column_name), 0] = \
                    float(readout)
            elif item_format != ValueType_skip:
                raise ValueError("Invalid format %s" % item_format)
            item.features.update(features)
        return item
    
    cpdef unsigned int get_features_count(self):
        return self.hash_function.hash_size
                
        
