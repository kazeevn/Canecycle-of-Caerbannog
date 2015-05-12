Required python packages
=======================
numpy
spooky
tables

Installation
============
sudo python setup.py install

Tests
=====
python setup.py build_ext --inplace
nosetests .

Usage - commad line
===================
usage: ccc.py [-h] [-l LEARN] [-c CACHE] [-H HOLDOUT] [-b HASH_SIZE]
              [--passes PASSES] [-p PREDICT] [-o OUTPUT] [--l1 L1] [--l2 L2]
              [--alpha ALPHA] [--beta BETA] [-v] [-d] [--discard-numeric]

An FTRL-based online learning machine

optional arguments:
  -h, --help            show this help message and exit
  -l LEARN, --learn LEARN
                        Input file for training in SHAD-LSML format
  -c CACHE, --cache CACHE
                        Use the cache file
  -H HOLDOUT, --holdout HOLDOUT
                        Each h-th line is not used for learning. After
                        learning, the average loss over h-th lines is
                        calculated
  -b HASH_SIZE, --hash-size HASH_SIZE
                        Hash table size in bits
  --passes PASSES       Number of passes on the learning data
  -p PREDICT, --predict PREDICT
                        Input files for prediction in SHAD-LSML format
  -o OUTPUT, --output OUTPUT
                        File to write the predictions into
  --l1 L1
  --l2 L2
  --alpha ALPHA         FTRL alpha
  --beta BETA           FTRL beta
  -v, --verbose
  -d, --debug
  --discard-numeric     Discard numeric features

Usage - Python API
==================
The system consists of several independent components -
if so desired you can, for example, supply your own optimizer
or data source.

Classes descriptions
--------------------
1. Item. Represenation of object. Properties:
   label - numpy.int, item label, -1 or +1
   indexes - numpy.ndarray, indexes of
      features with non-zero values. Usually you want
      them to be hashes.
   data - numpy.ndarray, feature values under item.indexes
   weight - the weight. Subject to imterpretation by
     optimization algorithms.
2. Source. Something that feeds data into the system. Currently, two
   types are implemented: Reader to read SHAD-LSML and
   CacheReader to read HDF5-based .can files
   Must support:
   restart(int holdout) - restart the source to the beginning,
      specifies holdout to use
   __iter__() - returnes iterator of Items
3. HashFunction - hash function to hash strings into uint64.
4. LossFunction - loss function, currently logistic loss is
   implemented. Must support:
   predict(item, weights)
   predict_proba(item, weights)
   get_loss(item, weights)
   get_gradient(item, weights)
5. WeightManager - reweights items, so that rare classes get more
   weight
6. Optimizer. Optimization algorithm implementaion.
   Currenly FRTL-proximal is implemented. Must support:
   step(item, weights) - feed Item into the algorith update weights
7. Classifier - Class that puts all above together.

Example 
-------
```python
from canecycle.weight_manager import WeightManager
from canecycle.loss_function import LossFunction
from canecycle.optimizer import Optimizer
from canecycle.hash_function import HashFunction
from canecycle.reader import from_shad_lsml
from canecycle.classifier import Classifier

hash_size = 2**20
source = from_shad_lsml("canecycle/tests/train_except.txt",
			hash_size=hash_size,
			discard_numeric=True,
			cache_file_name="/tmp/train_except.can")
optimizer = Optimizer(1e-10, 1e-10, feature_space_size=hash_size,
	    	      alpha=1e-4, beta=1e-3, loss_function=LossFunction())
classifier = Classifier(optimizer, LossFunction(), WeightManager(),
	     	       store_progressive_validation=False,
		       holdout=10,	
		       pass_number=3,
		       display=True,
		       use_cache=True)
classifier.fit(source)
source.close()
```