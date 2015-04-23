# cython: profile=True
from canecycle.source import NotInitialized
from canecycle.parser import read_shad_lsml_header
from canecycle.hash_function cimport HashFunction

# Should be @staticmethod, but Cython doesn't support it in cpdef
cpdef Reader from_shad_lsml(str filename, unsigned int hash_size):
    cdef list format
    cdef Parser parser
    cdef Reader reader
    format_ = read_shad_lsml_header(filename)
    hash_function = HashFunction(hash_size)
    parser = Parser(hash_function, format_)
    reader = Reader(filename, parser, 1)
    return reader


cdef class Reader(Source):
    def __cinit__(self, str filename, Parser parser, unsigned int skip):
        self.is_ready = False
        self.parser = parser
        self.skip = skip
        self.filename = filename
        
    def __iter__(self):
        return self
    
    cpdef restart(self, int holdout):
        """Restarts the source. Specify positive holdout to omit each h-th
        item negative to omit each but h-th item, zero to omit
        nothing

        """
        if self.is_ready:
            self.file.close()
        else:
            self.is_ready = True
        self.file = open(self.filename)
        for _ in xrange(self.skip):
            next(self.file)
        self.holdout = holdout
        self.holdout_counter = 0
            
    def __next__(self):
        cdef str line
        if not self.is_ready:
            raise NotInitialized("You should call restart before "
                                 "using a source")
        if self.holdout == 1:
            raise StopIteration()

        line = next(self.file)
        self.holdout_counter += 1
            
        if self.holdout > 0:
            if self.holdout_counter % self.holdout == 0:
                line = next(self.file)
                self.holdout_counter = 1
        elif self.holdout < 0:
            while self.holdout_counter % -self.holdout != 0:
                line = next(self.file)
                self.holdout_counter += 1
            self.holdout_counter = 0
        return self.parser.parse(line)
    
    cpdef unsigned int get_features_count(self):
        return self.parser.get_features_count()
