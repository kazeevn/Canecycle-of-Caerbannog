from itertools import izip
from canecycle.item import Item

def name_iterator():
    i = 0
    while True:
        yield "%d_" % i

cdef str transform_header(header_item):
    if header_item == 'CLICK':
        return 'label'
    elif header_item.startswith('CAT'):
        return 'categorial'
    elif header_item.startswith('NUM'):
        return 'numerical'
    else:
        raise ValueError("Unknown item in header %s" % header_item)

cdef list read_shad_lsml_header(filename):
    global transform_header
    input_file = open(filename)
    header = input.readline().split(',')
    # candy cane!
    result = ['label', 'categorial','numerical']
    #result = map(transform_header, header)
    input_file.close()
    return result


cdef class Reader(object):
    def __cinit__(self, hash_function):
        self.hash_function = hash_function
        
    def open(self, filename, format_, skip):
        self.filename = filename
        self.format = format_
        self.file = open(self.filename)
        self.skip = skip
        self.restart()
        
    cdef void restart(self):
        self.file.close()
        self.file = open(self.filename)
        for _ in xrange(self.skip):
            self.file.next()
    
    cdef unsigned int hash_size(self):
        return self.hash_function.hash_size
    
    def __next__(self):
        # StopIteration will rise through stack
        #TODO: init cycle variables
        line = file.next().split(',')
        item = Item(self.hash_function.hash_size)
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
        
        
