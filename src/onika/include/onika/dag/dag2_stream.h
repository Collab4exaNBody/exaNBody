/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
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

