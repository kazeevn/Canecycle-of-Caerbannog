#ifndef LOSS_FUNCTION_H_
#define LOSS_FINCTION_H_

#include "./item.h"
#include "./types.h"

class LossFunction {
 public:
  LossFunction(const float l1, const float l2) {}
  const float get_decision(const Item& item, const Weights& weights) {return 0.;}
  const float get_loss(const Item& item, const Weights& weights) {return 0;}
  // Do we need it? Do we need logistic loss internally?
  const Weights get_grad(const Item& item, const Weights& weight) {return Weights();}
};

#endif  //  LOSS_FUNCTION_H_
