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
    struct Dag2DotConfig
    {
      std::function< oarray_t<size_t,Nd>(size_t) > coord_func = nullptr;
      std::function< bool(size_t i) > mask_func = nullptr;
      double gw = 0.0; // grid to wave position fading
      std::pair<double,double> bbenlarge = { 0.0 , 0.0 }; // bounding box enlargment
      std::pair<double,double> urenlarge = { 0.0 , 0.0 }; // bounding box extra enlargment for upper right
      int grainsize = 1;
      bool fdp = false;
      bool add_legend = true;
      bool add_bounds_corner = false;
      bool movie_bounds = false;
      bool wave_group = false;
    };

    template<size_t Nd>
    std::ostream& dag_to_dot(
      const WorkShareDAG2<Nd>& dag ,
      const oarray_t<size_t,Nd>& domain ,
      std::ostream& out ,
      Dag2DotConfig<Nd> && config );

  }
}

