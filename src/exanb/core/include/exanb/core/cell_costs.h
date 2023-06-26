#pragma once

#include <exanb/core/basic_types.h>
#include <vector>

namespace exanb
{

  struct CellCosts
  {
    GridBlock m_block;
    std::vector<double> m_costs;
  };

}

