#include <gtest/gtest.h>
#include <math.h>

#include "./loss_function.h"
#include "./item.h"
#include "./types.h"

template <typename T>
boost::numeric::ublas::vector<T> vector_from_list(std::initializer_list<T> list) {
  boost::numeric::ublas::vector<T> result(list.size());
  for (auto& item : list) {
    unsigned int index = &item - list.begin();
    result[index] = item;
  }
  return result;
}
// http://stackoverflow.com/questions/27459318/how-to-create-a-nonempty-boost-ublasvector-inside-an-initializer-list

TEST(LossFunctionTest, Simple2DDecisionNoRegularization) {
  LossFunction loss_function(0., 0.);
  Item x = {1.0, 1, {1, 0}};
  Weights weights(vector_from_list<float>({0.2f, 0.8f}));
  ASSERT_FLOAT_EQ(0.2, loss_function.get_decision(x, weights));
  // proba equals to 1. / (1. + exp(-0.2)). Shall we return it or mimic VW?
  ASSERT_FLOAT_EQ(-log(1. / (1. + exp(-0.2))),
                  loss_function.get_loss(x, weights));
}
