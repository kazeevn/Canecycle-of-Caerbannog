#ifndef READER_H_
#define READER_H_

#include <iostream>

#include "./item.h"

class Reader {
 public:
  Reader(
      const HashFunction& hash_function,
      const unsigned int features_count);
  void open(std::istream input_stream);
  // TODO(kazeevn) make something fancy iterator-like
  Item get_item();
  bool has_items();
}

#endif  // READER_H_
