cimport numpy as np

class NotInitialized(BaseException):
    pass


cdef class Source(object):
    def __iter__(self):
        if not self.is_ready:
            raise NotInitialized("You should call restart before "
                                 "using a source")
        return self.iterator

    
def dropping_iterator(iterator, holdout):
    cdef np.uint64_t counter = 0
    for item in iterator:
        counter += 1
        if holdout > 0 and (counter - 1) % holdout == 0:
            continue
        elif holdout < 0 and (counter - 1) % -holdout != 0:
            continue
        yield item
