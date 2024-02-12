#pragma once

#include <iostream>

namespace onika
{

  namespace parallel
  {
    inline std::ostream& log_err() { return std::cerr; }
    inline std::ostream& log_out() { return std::cout; }
  }

}

