#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <boost/numeric/ublas/vector.hpp>
#include <iostream>

#include "./item.h"
#include "./types.h"

class Optimizer {
 public:
  Optimizer(
      const LossFunction& loss_function,
      std::ostream log);
  void step(
      const Item& item,
      const unsigned int64 step_number,
      Weights* weights);
}

#endif   // OPTIMIZER_H_
