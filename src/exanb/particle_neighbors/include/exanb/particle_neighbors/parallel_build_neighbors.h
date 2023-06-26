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

