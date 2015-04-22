from itertools import izip, imap
from canecycle.item import Item
import string

def name_iterator():
    cdef unsigned int i
    i = 0
    while True:
        yield "%d_" % i

#TODO(kazeevn) switch to proper Cython
ValueType_label = 0
ValueType_categorical = 1
ValueType_numerical = 2

def transform_header(header_item):
    if header_item == 'CLICK':
        return ValueType_label
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


class Reader:
    def __init__(self, hash_function, filename, format_, skip):
        self.hash_function = hash_function
        self.filename = filename
        self.format = format_
        self.file = open(self.filename)
        self.skip = skip
        self.restart()
        
    def restart(self):
        self.file.close()
        self.file = open(self.filename)
        for _ in xrange(self.skip):
            self.file.next()
    
    def __iter__(self):
        for line in self.file:
            processed_line = map(string.strip, line.split(','))
            item = Item(self.hash_function.hash_size)
            column_namer = name_iterator()
            for item_format, readout, column_name in \
                izip(self.format, processed_line, column_namer):
                if readout == '':
                    continue
                if item_format == ValueType_label:
                    item.label = int(readout)
                elif item_format == ValueType_categorical:
                    item.features[self.hash_function.hash(
                        column_name + readout)] = 1
                elif item_format == ValueType_numerical:
                    item.features[self.hash_function.hash(column_name)] = \
                        float(readout)
                else:
                    raise ValueError("Invalid format %s" % item_format)
            yield item
        
        
