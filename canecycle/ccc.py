#!/usr/bin/env python2


import argparse
import numpy as np

from canecycle.hash_function import HashFunction
from canecycle.reader import from_shad_lsml
from canecycle.classifier import Classifier
from canecycle.weight_manager import WeightManager
from canecycle.optimizer import Optimizer
from canecycle.loss_function import LossFunction
from canecycle.cache import CacheReader


def check_negative(value):
    int_value = int(value)
    if int_value < 0:
         raise argparse.ArgumentTypeError("%s isn't a valid non-negative integer" % value)
    return int_value


def main():
    parser = argparse.ArgumentParser(description="An FTRL-based online learning machine")
    parser.add_argument("-l", "--learn", type=str,
                        help="Input file for training in SHAD-LSML format")
    parser.add_argument("-c", "--cache", type=str,
                        help="Use the cache file")
    parser.add_argument("-H", "--holdout", type=check_negative, default=0,
                        help="Each h-th line is not used for learning. After learning, "
                        "the average loss over h-th lines is calculated")
    parser.add_argument("-b", "--hash-size", type=int,
                        help="Hash table size in bits")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of passes on the learning data")
    parser.add_argument("-p", "--predict", type=str,
        help="Input files for prediction in SHAD-LSML format")
    parser.add_argument("-o", "--output", type=str,
        help="File to write the predictions into")
    parser.add_argument("--l1", type=float, default=0)
    parser.add_argument("--l2", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1e-5,
                        help="FTRL alpha")
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="FTRL beta")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--discard-numeric", action="store_true",
                        help="Discard numeric features")
    
    args = parser.parse_args()
    if args.debug:
        args.verbose = True
        np.seterr(invalid='raise', over='raise')
    
    if args.predict and not args.output:
        parser.error("--predict requires --output")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    loss_function = LossFunction()
    cache_file_name = None
    if args.learn:
        if not args.hash_size:
            parser.error("Hash size must be specified if reading from"
                         " text file")
        hash_size = 2**args.hash_size
        if args.cache:
            source = from_shad_lsml(args.learn, hash_size, args.discard_numeric, args.cache)
        else:
            source = from_shad_lsml(args.learn, hash_size, args.discard_numeric, '')
    elif args.cache:
        if args.discard_numeric:
            parser.error("Can't skip numerics in cache files")
        source = CacheReader(args.cache)
        if args.hash_size and 2**args.hash_size != source.get_features_count():
            parser.error("Specified hash size differs from one in the"
                         " cache file")
        hash_size = source.get_features_count()
    else:
        parser.error("Must specify either --learn of --cache")

    optimizer = Optimizer(args.l1, args.l2, hash_size, args.alpha,
                          args.beta, loss_function)

    classifier = Classifier(optimizer, loss_function, WeightManager(),
                            False, args.holdout, args.passes,
                            display=args.verbose, use_cache=bool(args.cache and args.learn))
    
    classifier.fit(source)
    source.close()
    if args.holdout != 0:
        print("Average holdout loss: %f" % classifier.get_holdout_loss())

    
    if args.predict:
        predict_file = from_shad_lsml(
            args.predict, hash_size, args.discard_numeric)
        output_file = open(args.output, 'w')
        for prediction in classifier.predict_proba(predict_file):
            output_file.write("%g\n" % prediction)
        output_file.close()

if __name__ == '__main__':
    main()
