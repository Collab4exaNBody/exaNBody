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

