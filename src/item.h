#ifndef ITEM_H_
#define ITEM_H_

#include <boost/numeric/ublas/vector_sparse.hpp>

typedef boost::numeric::ublas::mapped_vector<int> Features;

struct Item {
  float weight;
  int label;  // +1; -1
  Features features;
};

#endif  // ITEM_H_
