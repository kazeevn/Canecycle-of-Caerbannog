#ifndef CLASSIFIER_H_
#define CALSSIFIER_H_

#include <string>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>

#include "loss_function.h"
#include "reader.h"

class Classifier {
public:
  Classifier(
      const bool enable_progressive_validation,
      const unsigned int holdout,  // Hold each n-nth sample for validation. 0 disables the feature.
      const unsigned int passes,
      const LossFunction& loss_function,
      const Optimizer& optimizer,
      Reader& reader,
      std::ofstream log);

  void fit(const string& file_name);
  const double get_average_progressive_loss();
  const double get_average_holdout_loss();
  boost::numeric::ublas::vector<double> predict(const Item& item);
  void predict(const string& input_file_name, const string& output_file_name);
};
#endif  // CLASSIFIER_H_
