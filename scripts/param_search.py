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
