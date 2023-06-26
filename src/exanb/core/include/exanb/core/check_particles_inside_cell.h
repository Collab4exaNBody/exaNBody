#pragma once

#include <exanb/core/log.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/geometry.h>

namespace exanb
{

  template<class GridT>
  inline bool check_particles_inside_cell(GridT& grid, bool exclude_ghosts=true, bool abort_upon_failure=false)  
  {
    size_t n_cells = grid.number_of_cells();
    const double threshold = grid.epsilon_cell_size2();
    size_t ghost_layers = grid.ghost_layers();
    IJK dims = grid.dimension();
    
    for(size_t cell_i=0; cell_i<n_cells; cell_i++)
    {
      IJK cell_loc = grid_index_to_ijk(dims,cell_i);
      bool is_ghost = inside_grid_shell(dims,0,ghost_layers,cell_loc);

      if( !is_ghost || !exclude_ghosts )
      {
        AABB bounds = grid.cell_bounds(cell_loc);
        size_t n_particles = grid.cell_number_of_particles(cell_i);
        for(size_t p=0; p<n_particles; p++)
        {
          Vec3d r = grid.particle_position( cell_i , p );
          bool inside = is_inside(bounds,r);
          double rel_dist = 0.;
          if( ! inside )
          {
            rel_dist = min_distance2_between(r,bounds);
            if( rel_dist > threshold )
            {
	            lerr <<"Warning: in cell #"<<cell_i<<" at "<<cell_loc<<" (bounds="<<grid.cell_bounds(cell_loc)<<") : particle "<<p<<" at "<<r<<" outside cell : d^2 = "<<rel_dist<<", threshold="<<threshold << std::endl;
	            if( abort_upon_failure ) { std::abort(); }
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  
}

