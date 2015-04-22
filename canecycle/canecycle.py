#!/usr/bin/env python2
import argparse
from canecycle.parser import read_shad_lsml_header, Parser
from canecycle.hash_function import HashFunction
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
    parser = argparse.ArgumentParser(descripption="An FTRL-based online learning machine")
    parser.add_argument("-l", "--learn", type=str,
                        help="Input file for training in SHAD-LSML format")
    parser.add_argument("-h", "--holdout", type=check_negative, default=0,
                        help="Each h-th line is not used for learning. After learning,
                        the average loss over h-th lines is calculated")
    parser.add_argument("-b", "--hash-size", type=int,
                        help="Hash table size in bits")
    parser.add_argument("--progressive", type="store_true",
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

    hash_function = HashFunction(args.b)
    classifier = Classifier(Optimizer(), LossFunction(), WeightManager(),
                            args.progressive, args.holdout, args.passes)
    
    format_ = read_shad_lsml_header(args.learn)
    parser = Parser(hash_function, format_)
    classifier.fit(args.learn, parser)
    if args.holdout != 0:
        print("Average holdout loss: %f" % classifier.hold_out)
    # TODO(kazeevn) add progressive_validation
    
    if args.predict:
        format_ = read_shad_lsml_header(args.learn)
        parser = Parser(hash_function, format_)
        calssifier.predict(args.predict, parser, args.output)
        

if __name__ == '__main__':
    main()
