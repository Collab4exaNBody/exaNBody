#pragma once

#include <iostream>

namespace onika
{
  namespace task
  {
    // undefined task functor
    struct NullTaskFunctor
    {
      template<class... T> inline void operator () (T...) { std::cerr<<"Fatal error: empty kernel functor called\n"; std::abort(); }
    };

  }
  
}


