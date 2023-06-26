#pragma once

#include <cstdlib>
#include <onika/oarray.h>
#include <iostream>
#include <onika/dag/dag.h>

namespace onika
{
  namespace dag
  {
  
    template<size_t Nd>
    std::ostream& dag_to_dot(
      const WorkShareDAG<Nd>& dag ,
      const oarray_t<size_t,Nd>& domain ,
      std::ostream& out ,
      double position_scramble = 0.0 ,
      int grainsize = 1,
      bool fdp = false ,
      std::function< oarray_t<size_t,Nd>(size_t) > coord_func = nullptr ,
      std::function< bool(const oarray_t<size_t,Nd>& c) > mask_func = nullptr );

  }
}

