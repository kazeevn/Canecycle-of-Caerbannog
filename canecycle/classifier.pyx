# cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np
#TODO(shiryaev): same c-types in all project files
from cpython cimport bool as c_bool

from canecycle.item cimport Item
from canecycle.reader cimport Reader
from canecycle.weight_manager cimport WeightManager
#TODO(shiryaev) import->cimport?
from canecycle.optimizer import Optimizer
from canecycle.loss_function cimport LossFunction

#TODO(shiryaev) save/load model
cdef class Classifier(object):
    cdef WeightManager weight_manager
    cdef optimizer
    cdef LossFunction loss_function
    
    cdef c_bool store_progressive_validation
    cdef unsigned int pass_number
    cdef unsigned long items_processed
    cdef unsigned long validation_index
    cdef np.ndarray weights
    
    cdef int holdout
    cdef unsigned long holdout_items_processed
    cdef list progressive_validation_loss # optimizer loss
    cdef float average_training_loss # optimizer loss
    cdef float holdout_loss # lossfunc loss
    
    cdef c_bool display
    
    def __cinit__(self, optimizer, loss_function, WeightManager weight_manager,
            c_bool store_progressive_validation, int holdout, int pass_number,
            c_bool display=False):
        
        if pass_number < 0:
            raise ValueError("Negative number of passes.")
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.weight_manager = weight_manager
        self.store_progressive_validation = store_progressive_validation
        self.holdout = holdout
        self.pass_number = pass_number
        self.display = display
    
    cdef int predict_item(self, Item item):
        return self.loss_function.get_decision(item, self.weights)
    
    cdef float predict_proba_item(self, Item item) except *:
        return self.loss_function.get_loss(item, self.weights)
    
    def predict(self, Reader reader):
        #TODO(shiryaev): display
        cdef Item item
        reader.restart(0)
        for item in reader:
            yield self.predict_item(item)
    
    def predict_proba(self, Reader reader):
        cdef Item item
        reader.restart(0)
        for item in reader:
            yield self.predict_proba_item(item)
    
    cdef void run_holdout_pass(self, Reader reader):
        cdef Item item
        for item in reader:
            #TODO(shiryaev): display progress
            self.holdout_loss += self.predict_proba_item(item)
            self.holdout_items_processed += 1
    
    cdef void run_train_pass(self, Reader reader):
        cdef Item item
        for item in reader:
            # validation routine
            if self.items_processed == self.validation_index - 1:
                self.validation_index *= 2
                if self.store_progressive_validation:
                    self.progressive_validation_loss.append(
                        self.predict_proba_item(item))
                if self.display:
                    #TODO(shiryaev): display progress
                    print self.items_processed,
                    print self.average_training_loss / self.items_processed
            
            item.weight = self.weight_manager.get_weight(item.label, item.weight)
            self.average_training_loss += self.predict_proba_item(item)
            self.weights = self.optimizer.step(item, self.items_processed, self.weights)
            self.items_processed += 1
    
    cpdef fit(self, Reader reader, continue_fitting=False):
        if not continue_fitting:
            self.weights = np.zeros(reader.get_features_count(), dtype=float)    
            self.items_processed = 0
            self.holdout_items_processed = 0
            self.validation_index = 1
            self.average_training_loss = 0.
            self.holdout_loss = 0.
        
        cdef unsigned int pass_index
        for pass_index in range(self.pass_number):
            reader.restart(self.holdout)
            self.run_train_pass(reader)
        
        self.average_training_loss /= self.items_processed
        
        # holdout pass
        if self.holdout != 0:
            reader.restart(-self.holdout)
            self.run_holdout_pass(reader)
            self.holdout_loss /= self.holdout_items_processed
    
    cpdef list get_progressive_validation(self):
        return self.progressive_validation_loss
    
    cpdef double get_average_loss(self):
        return self.average_training_loss
    
    cpdef double get_holdout_loss(self):
        return self.holdout_loss

