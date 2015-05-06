#!/usr/bin/env python2
import argparse
from canecycle.hash_function import HashFunction
from canecycle.reader import from_shad_lsml
from canecycle.classifier import Classifier
from canecycle.weight_manager import WeightManager
from canecycle.optimizer import Optimizer
from canecycle.loss_function import LossFunction

def check_negative(value):
    int_value = int(value)
    if int_value < 0:
         raise argparse.ArgumentTypeError("%s isn't a valid non-negative integer" % value)
    return int_value

def main():
    parser = argparse.ArgumentParser(description="An FTRL-based online learning machine")
    parser.add_argument("-l", "--learn", type=str, required=True,
                        help="Input file for training in SHAD-LSML format")
    parser.add_argument("-H", "--holdout", type=check_negative, default=0,
                        help="Each h-th line is not used for learning. After learning, "
                        "the average loss over h-th lines is calculated")
    parser.add_argument("-b", "--hash-size", type=int, required=True,
                        help="Hash table size in bits")
    parser.add_argument("--progressive", action="store_true",
                        help="Enables output of progressive validation")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of passes on the learning data")
    parser.add_argument("-p", "--predict", type=str,
        help="Input files for prediction in SHAD-LSML format")
    parser.add_argument("-o", "--output", type=str,
        help="File to write the predictions into")

    args = parser.parse_args()
    if args.predict and not args.output:
        parser.error("--predict requires --output")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    loss_function = LossFunction()
    optimizer = Optimizer(0, 0, 2**args.hash_size, 1e-5, 1e-4, loss_function)
    classifier = Classifier(optimizer, loss_function, WeightManager(),
                            args.progressive, args.holdout, args.passes, display=True)

    reader = from_shad_lsml(args.learn, args.hash_size)
    classifier.fit(reader)
    if args.holdout != 0:
        print("Average holdout loss: %f" % classifier.get_holdout_loss())
    # TODO(kazeevn) add progressive_validation
    
    if args.predict:
        predict_file = from_shad_lsml(args.predict, args.hash_size)
        output_file = open(args.output, 'w')
        for prediction in classifier.predict_proba(predict_file):
            output_file.write("%g\n" % prediction)
        output_file.close()

if __name__ == '__main__':
    main()
