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

#include <exanb/core/grid.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/parallel_build_neighbors_impl.h>

namespace exanb
{
  

  /*
   * populate GridParticleNeighbors stores bonds (pairs of particles) for each cell C, such that :
   * for each particle Pa in C, we have a list of bonds Bi that connect Pa to Pb such that Pb is "after" Pa.
   * this "after" order is given by the lexicographical order (cell,particle)
   * 
   * must be called from a sequential region.
   */
  template<typename GridT>
  static inline void parallel_build_neighbors(GridT& grid, AmrGrid& amr, GridParticleNeighbors& pb, double max_dist)
  {
    parallel_build_neighbors_impl<GridT,false> ( grid, amr, pb, max_dist );
  }

} // namespace exanb

