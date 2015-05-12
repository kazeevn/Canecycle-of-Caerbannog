#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


import numpy as np
cimport numpy as np
from cpython cimport bool

from canecycle.item cimport Item
from canecycle.source cimport Source
from canecycle.weight_manager cimport WeightManager
from canecycle.loss_function cimport LossFunction
from canecycle.optimizer cimport Optimizer


cdef class Classifier(object):
    cdef WeightManager weight_manager
    cdef Optimizer optimizer
    cdef LossFunction loss_function
    
    cdef bool store_progressive_validation
    cdef np.uint64_t pass_number
    cdef np.uint64_t items_processed
    cdef np.uint64_t training_validation_index
    cdef np.ndarray weights
    
    cdef np.int_t holdout
    cdef np.uint64_t holdout_items_processed
    cdef np.uint64_t holdout_validation_index
    cdef list progressive_validation_loss
    cdef np.float_t average_training_loss
    cdef np.float_t holdout_loss
    
    cdef bool display
    cdef np.uint64_t max_iteration
    
    cdef bool use_cache
    
    def __cinit__(self, Optimizer optimizer, LossFunction loss_function,
                  WeightManager weight_manager, bool store_progressive_validation,
                  np.int_t holdout, np.uint64_t pass_number, bool display=False,
                  np.uint64_t max_iteration=1000000000, bool use_cache=False):
        
        if pass_number < 0:
            raise ValueError("Negative number of passes.")
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.weight_manager = weight_manager
        self.store_progressive_validation = store_progressive_validation
        self.progressive_validation_loss = []
        self.holdout = holdout
        self.pass_number = pass_number
        self.display = display
        self.max_iteration = max_iteration
        self.use_cache=use_cache
    
    cdef np.int_t predict_item(self, Item item):
        return self.loss_function.get_decision(item, self.weights)
    
    cdef np.float_t predict_proba_item(self, Item item):
        return self.loss_function.get_loss(item, self.weights)
    
    def predict(self, Source reader):
        cdef Item item
        reader.restart(0)
        for item in reader:
            yield self.predict_item(item)
    
    def predict_proba(self, Source reader):
        cdef Item item
        reader.restart(0)
        for item in reader:
            yield self.predict_proba_item(item)

    cdef void run_train_pass(self, Source reader):
        cdef Item item
        for item in reader:
            item_loss = self.loss_function.get_loss(item, self.weights)
            self.average_training_loss += item_loss

            if self.items_processed == self.training_validation_index - 1:
                self.training_validation_index *= 2

                if self.store_progressive_validation:
                    self.progressive_validation_loss.append(item_loss)
                if self.display:
                    print ('%15d %15.6e %15.6e' % (
                        self.items_processed,
                        self.average_training_loss / self.items_processed,
                        item_loss))
            
            item.weight = self.weight_manager.get_weight(item.label, item.weight)
            self.weights = self.optimizer.step(item, self.weights)
            self.items_processed += 1
        if self.display:
            print ('%15d %15.6e %15.6e' % (
                self.items_processed,
                self.average_training_loss / self.items_processed,
                item_loss))

    
    cdef void run_holdout_pass(self, Source reader) except *:
        cdef Item item
        for item in reader:
            if self.holdout_items_processed == self.holdout_validation_index - 1:
                self.holdout_validation_index *= 2
                if self.display:
                    print ('%15d %15.6e %15.6e' % (
                        self.holdout_items_processed,
                        self.holdout_loss / self.holdout_items_processed,
                        self.predict_proba_item(item)))
            self.holdout_loss += self.loss_function.get_loss(item, self.weights)
            self.holdout_items_processed += 1
        if self.display:
             print ('%15d %15.6e %15.6e' % (
                 self.holdout_items_processed,
                 self.holdout_loss / self.holdout_items_processed,
                 self.predict_proba_item(item)))
    
    cpdef fit(self, Source reader, bool continue_fitting=False):
        if self.display:
            print '{:-^50}'.format(' TRAINING ')
            print ('%15s %15s %15s' % ('iteration', 'average', 'last'))
            print ('%15s %15s %15s' % ('number', 'loss', 'loss'))
        if not continue_fitting:
            self.weights = np.zeros(reader.get_features_count(), dtype=np.float_)     
            self.items_processed = 0
            self.holdout_items_processed = 0
            self.training_validation_index = 1
            self.holdout_validation_index = 1
            self.average_training_loss = 0.
            self.holdout_loss = 0.
        
        cdef np.uint64_t pass_index
        for pass_index in range(self.pass_number):
            if self.use_cache:
                if pass_index == 0:
                    reader.restart(0, write_cache=True)
                else:
                    reader.restart(0, use_cache=True)
            else:
                reader.restart(self.holdout)
            if self.items_processed < self.max_iteration:
                self.run_train_pass(reader)
        
        self.average_training_loss /= self.items_processed
        
        if self.holdout != 0:
            if self.display:
                print '{:-^50}'.format(' HOLDOUT PASS ')
            if self.use_cache:
                reader.restart(-self.holdout, use_cache=True)
            else:
                reader.restart(-self.holdout)
            self.run_holdout_pass(reader)
            self.holdout_loss /= self.holdout_items_processed
    
    cpdef list get_progressive_validation(self):
        return self.progressive_validation_loss
    
    cpdef np.float_t get_average_loss(self):
        return self.average_training_loss
    
    cpdef np.float_t get_holdout_loss(self):
        return self.holdout_loss
