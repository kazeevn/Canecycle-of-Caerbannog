from itertools import izip
from canecycle.item import Item

def name_iterator():
    i = 0
    while True:
        yield "%d_" % i

cpdef string transform_header(header_item):
    if header_item == 'CLICK':
        return 'label'
    elif header_item.startswith('CAT'):
        return 'categorial'
    elif header_item.startswith('NUM'):
        return 'numerical'
    else:
        raise ValueError("Unknown item in header %s" % header_item)

def read_shad_lsml_header(filename):
    input_file = open(filename)
    header = input.readline.split(',')
    result =  map(transform_header, header)
    input_file.close()
    return result


cydef class Reader:
    def __cinit__(self, hash_function):
        self.hash_function = hash_function
        
    def open(filename, format_, skip):
        self.filename = filename
        self.format = format_
        self.file = open(self.filename)
        self.skip = skip
        self.restart()
        
    def restart():
        self.file.close()
        self.file = open(self.filename)
        for _ in xrange(self.skip):
            self.file.next()
    
    def __next__():
        # StopIteration will rise through stack
        line = file.next().split(',')
        item = Item(hash_function.hash_size)
        column_namer = name_iterator()
        for item_format, readout, column_name in \
            izip(self.format, line, column_namer):
            if item_format == 'label':
                item.label = int(readout)
            elif item_format == 'categorial':
                item.features[self.hash_function.hash(
                    column_name + readout)] = 1
            elif item_format == 'numerical':
                item.features[self.hash_function.hash(column_name)] = \
                    float(readout)
            else:
                raise ValueError("Invalid format %s" % item_format)
        return item
        
        
