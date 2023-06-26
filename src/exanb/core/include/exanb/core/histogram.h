#pragma once

#include <vector>
#include <limits>

namespace exanb
{

  template<typename ValueType = double >
  struct Histogram
  {
    ValueType m_min_val = std::numeric_limits<ValueType>::max();
    ValueType m_max_val = std::numeric_limits<ValueType>::lowest();
    std::vector<ValueType> m_data;
  };

}


