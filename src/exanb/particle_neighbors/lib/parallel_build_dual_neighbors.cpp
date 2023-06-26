#include <exanb/particle_neighbors/parallel_build_dual_neighbors.h>

#include <exanb/core/particle_id_codec.h>
#include <exanb/core/algorithm.h>

#include <atomic>
#include <algorithm>
#include <functional>

namespace exanb
{

  void parallel_build_dual_neighbors(const GridParticleNeighbors& primary, GridParticleNeighbors& dual)
  {
    size_t n_cells = primary.size();
    dual.resize( n_cells );
    
#   pragma omp parallel
    {

#     pragma omp for
      for(size_t i=0;i<n_cells;i++)
      {
        dual[i].nbh_start.assign( primary[i].nbh_start.size() , 0 );
      }

#     pragma omp for
      for(size_t cell_a=0;cell_a<n_cells;cell_a++)
      {
        size_t n_particles = primary[cell_a].nbh_start.size();
        for(size_t p_a=0;p_a<n_particles;p_a++)
        {
          size_t neighbor_index = 0;
          if( p_a > 0 ) { neighbor_index = primary[cell_a].nbh_start[p_a-1]; }
          size_t neighbor_end = primary[cell_a].nbh_start[p_a];
          for(;neighbor_index<neighbor_end;neighbor_index++)
          {
            size_t cell_b=0, p_b=0;
            exanb::decode_cell_particle(primary[cell_a].neighbors[neighbor_index], cell_b, p_b);
#           pragma omp atomic
            ++ dual[cell_b].nbh_start[p_b];
          }
        }
      }
      
//#     pragma omp barrier // implicit after omp for

#     pragma omp for
      for(size_t cell_b=0;cell_b<n_cells;cell_b++)
      {
        ssize_t n_particles = dual[cell_b].nbh_start.size();
        size_t n_cell_neighbors = 0;
        if( n_particles > 0 ) { n_cell_neighbors = dual[cell_b].nbh_start[n_particles-1]; }
        exclusive_prefix_sum( dual[cell_b].nbh_start.data(), n_particles );
        if( n_particles > 0 ) { n_cell_neighbors += dual[cell_b].nbh_start[n_particles-1]; }
        dual[cell_b].neighbors.resize( n_cell_neighbors );
      }      
            
//#     pragma omp barrier // implicit after omp for

#     pragma omp for
      for(size_t cell_a=0;cell_a<n_cells;cell_a++)
      {
        size_t n_particles = primary[cell_a].nbh_start.size();
        for(size_t p_a=0;p_a<n_particles;p_a++)
        {
          size_t neighbor_index = 0;
          if( p_a > 0 ) { neighbor_index = primary[cell_a].nbh_start[p_a-1]; }
          size_t neighbor_end = primary[cell_a].nbh_start[p_a];
          for(;neighbor_index<neighbor_end;neighbor_index++)
          {
            size_t cell_b=0, p_b=0;
            exanb::decode_cell_particle(primary[cell_a].neighbors[neighbor_index], cell_b, p_b);
            // FIXME: performance is ok, but this has to be proven to be rock solid
            uint32_t bsindex = ( *(std::atomic<uint32_t>*)( dual[cell_b].nbh_start.data()+p_b ) ) ++ ;
            dual[cell_b].neighbors[ bsindex ] = exanb::encode_cell_particle(cell_a,p_a);
          }
        }
      }

      // final step : sort neighbor particles so that cell is decreasing and in each neighbor cell, neighbor index is increasing
#     pragma omp for
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        size_t n_particles = dual[cell_i].nbh_start.size();
        for(size_t p_j=0;p_j<n_particles;p_j++)
        {
          size_t neighbor_index = 0;
          if( p_j > 0 ) { neighbor_index = dual[cell_i].nbh_start[p_j-1]; }
          size_t neighbor_end = dual[cell_i].nbh_start[p_j];
          std::sort( dual[cell_i].neighbors.begin()+neighbor_index, dual[cell_i].neighbors.begin()+neighbor_end );
        }
      }

    } // omp parallel
    
  }

}

