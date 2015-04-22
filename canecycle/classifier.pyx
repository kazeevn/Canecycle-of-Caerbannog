from __future__ import division

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np

from canecycle.item import Item
from canecycle.reader import Reader
from canecycle.weight_manager import WeightManager
from canecycle.optimizer import Optimizer
from canecycle.loss_function import LossFunction


cdef class Classifier(object):
    cdef float average_loss
    cdef float holdout_loss
    cdef int holdout_items
    cdef weight_manager
    cdef optimizer
    cdef loss_function
    cdef list progressive_validation_loss
    cdef int holdout
    cdef int pass_number
    cdef int items_processed
    cdef weights
    cdef display
    
    def __cinit__(self, optimizer, loss_function, weight_manager,
            int get_progressive_validation, int holdout, int pass_number,
            display=False):
        
        if holdout < 0:
            raise ValueError("Negative holdout.")
        
        if pass_number < 0:
            raise ValueError("Negative number of passes.")
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.get_progressive_validation = get_progressive_validation
        self.holdout = holdout
        self.pass_number = pass_number
        self.weight_manager = weight_manager
        self.display = display
    
    
    cdef int predict_item(self, item):
        return self.optimizer.get_decision(item, self.weights)
    
    
    cdef float predict_proba_item(self, item):
        return self.optimizer.get_loss(item, self.weights)
    
    cdef void predict(self, reader, output_file):
        #TODO: display
        for item in reader:
            output_file.write(self.predict_item(item))
            output_file.write('\n')
    
    
    cdef void predict_proba(self, reader, output_file):
        #TODO: display
        for item in reader:
            output_file.write(self.predict_proba_item(item))
            output_file.write('\n')
    
    
    cdef void fit(self, reader):
        self.weights = coo_matrix(reader.hash_size(), dtype=np.float32)
        self.items_processed = 0
        self.holdout_items = 0
        cdef unsigned int current_pass
        cdef item = Item(reader.hash_size())
        cdef int validation_index = 1
        for current_pass in range(self.pass_number):
            for item in reader:
                
                item.weight = self.weight_manager.get_weight(item.label, item.weight)
                
                #validation routine
                if self.items_processed==validation_index-1:
                    validation_index *= 2
                    self.progressive_validation_loss.append(self.predict_item(item))
                    if self.display:
                        #TODO: display everything
                        print self.items_processed
                
                if self.holdout != 0 and self.items_processed%self.holdout == 0:
                    #holdout routine
                    self.holdout_loss += self.predict_item(item)
                    self.holdout_items += 1
                else:
                    # step routine
                    self.optimizer.step(item, self.weights)
                
                self.items_processed += 1
            reader.restart()
    
    
    #TODO
    def get_progressive_validation(self):
        return self.progressive_validation_loss
    
    
    def hold_out(self):
        return self.holdout_loss / self.holdout_items

