Installation
============
Required python packages: 
cython, numpy, spooky, tables

```bash
sudo python setup.py install
```

Tests
=====
```bash
python setup.py build_ext --inplace
nosetests .
```

Usage - command line
===================
```
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
```
Usage - Python API
==================
The system consists of several independent components -
if so desired you can, for example, supply your own optimization
algorithm or data source.

Classes descriptions
--------------------
1. Item. Representation of object. Properties:
   label - numpy.int, item label, -1 or +1
   indexes - numpy.ndarray, indexes of
      features with non-zero values. Usually you want
      them to be hashes.
   data - numpy.ndarray, feature values under item.indexes
   weight - the weight. Subject to interpretation by
     optimization algorithms.
2. Source. Something that feeds data into the system. Currently, two
   types are implemented: Reader to read SHAD-LSML and
   CacheReader to read HDF5-based .can files
   Must support:
   restart(int holdout) - restart the source to the beginning,
      specifies holdout to use
   __iter__() - returns iterator of Items
3. HashFunction - hash function to hash strings into uint64.
4. LossFunction - loss function, currently logistic loss is
   implemented. Must support:
   predict(item, weights)
   predict_proba(item, weights)
   get_loss(item, weights)
   get_gradient(item, weights)
5. WeightManager - reweights items, so that rare classes get more
   weight
6. Optimizer. Optimization algorithm implementation.
   Currently FRTL-proximal is implemented. Must support:
   step(item, weights) - feed Item into the algorithm update weights
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

Advanced Example
----------------
We can easily use any advanced algorithm in Python
for hyperparameters search. The following example is available in
"scripts/param_search.py" and can be run as following:
```bash
python scripts/param_search.py canecycle/tests/train_except_git.can
```
Code:
```python
from scipy.optimize import minimize
from argparse import ArgumentParser

from canecycle.optimizer import Optimizer
from canecycle.classifier import Classifier
from canecycle.weight_manager import WeightManager
from canecycle.cache import CacheReader
from canecycle.loss_function import LossFunction


def try_params(params, source):
    l1, l2, alpha, beta = params
    optimizer = Optimizer(
        l1, l2, source.get_features_count(), alpha, beta, LossFunction())
    classifier = Classifier(
        optimizer, LossFunction(), WeightManager(), False, 100, 2, False)
    classifier.fit(source)
    loss = classifier.get_holdout_loss()
    print params, loss
    return loss


def optimize(initial_guess, bounds, source, maxiter=10):
    return minimize(try_params, initial_guess, bounds=bounds,
                    args=source, options={'maxiter': maxiter, 'disp': True})

def main():
    parser = ArgumentParser(description="Finds optimal "
                            "hyperparameters for FTRL")
    parser.add_argument("input", type=str, nargs=1,
                        help="Input file in .can format")
    args = parser.parse_args()
    source = CacheReader(args.input[0])
    initial_guess = (1e-10, 1e-10, 1e-4, 1e-3)
    bounds = ((1e-20, 1), (1e-20, 1), (1e-10, 1), (1e-10, 1))
    print optimize(initial_guess, bounds, source)
        
if __name__ == '__main__':
    main()
```