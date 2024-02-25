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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/memory/allocator.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_iterator.h>
#include <exanb/particle_neighbors/flat_neighbor_lists.h>

namespace exanb
{
  template<typename GridT>
  class ChunkNeighbors2FlatParticleLists : public OperatorNode
  {
    ADD_SLOT( GridT              , grid            , INPUT , REQUIRED );
    ADD_SLOT( GridChunkNeighbors , chunk_neighbors , INPUT , REQUIRED );
    ADD_SLOT( double             , nbh_dist        , INPUT , 0.0 );
    ADD_SLOT( FlatPartNbhList    , flat_nbh_list   , INPUT_OUTPUT );

  public:
    inline void execute () override final
    {
      if( !grid.has_value() ) { return; }
      if( grid->number_of_particles()==0 ) { return; }
        
      ldbg << "cs="<< chunk_neighbors->m_chunk_size <<std::endl;

      IJK dims = grid->dimension();

      uint64_t total_particles = grid->number_of_particles();
      uint64_t total_nbh_count = 0;

      auto cells = grid->cells();
      using CellT = std::remove_cv_t< std::remove_reference_t< decltype(cells[0]) > >;
      ChunkParticleNeighborsIterator<CellT> chunk_nbh_it_in = { grid->cells() , chunk_neighbors->data() , dims , chunk_neighbors->m_chunk_size };
      const double nbh_d2 = (*nbh_dist) * (*nbh_dist) ;

      flat_nbh_list->m_neighbor_offset.assign( total_particles + 1 , 0 );
      auto * __restrict__ particle_nbh_count = flat_nbh_list->m_neighbor_offset.data();
      flat_nbh_list->m_half_count.assign( total_particles , 0 );
      auto * __restrict__ half_nbh_count = flat_nbh_list->m_half_count.data();
      
      const auto * __restrict__ cell_particle_offset = grid->cell_particle_offset_data();

#     pragma omp parallel
      {
        auto chunk_nbh_it = chunk_nbh_it_in;
        GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) )
        {
          // std::cout<<"dims="<<dims<<" cell_a="<<cell_a<<" loc_a="<<loc_a<<std::endl;
          const double* __restrict__ rx_a = cells[cell_a][field::rx];
          const double* __restrict__ ry_a = cells[cell_a][field::ry];
          const double* __restrict__ rz_a = cells[cell_a][field::rz];
          const size_t n_particles_a = cells[cell_a].size();
          chunk_nbh_it.start_cell( cell_a , n_particles_a );
          const bool is_ghost = grid->is_ghost_cell( loc_a );
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            chunk_nbh_it.start_particle( p_a );
            while( ! chunk_nbh_it.end() )
            {
              size_t cell_b=0, p_b=0;
              chunk_nbh_it.get_nbh( cell_b , p_b );
              const Vec3d dr = { cells[cell_b][field::rx][p_b] - rx_a[p_a] , cells[cell_b][field::ry][p_b] - ry_a[p_a] , cells[cell_b][field::rz][p_b] - rz_a[p_a] };
              if( norm2(dr) <= nbh_d2 )
              {
                auto part_idx = cell_particle_offset[cell_a] + p_a;
                ++ particle_nbh_count[ part_idx + 1 ];
                if( ssize_t(cell_b)<cell_a || ( ssize_t(cell_b)==cell_a && p_b<p_a ) ) ++ half_nbh_count[part_idx];
              }
              chunk_nbh_it.next();
            }
          }               
        }
        GRID_OMP_FOR_END
      }
      
      for(size_t i=1;i<=total_particles;i++) particle_nbh_count[i] += particle_nbh_count[i-1];
      total_nbh_count = particle_nbh_count[total_particles];
      ldbg << "total_particles="<<total_particles<<" , total_nbh_count="<<total_nbh_count<<std::endl;
      
      using ParticleIndex = typename FlatPartNbhList::ParticleIndex;
      flat_nbh_list->m_neighbor_list.assign( total_nbh_count , std::numeric_limits<ParticleIndex>::max() );
      auto * __restrict__ neighbor_list = flat_nbh_list->m_neighbor_list.data();

#     pragma omp parallel
      {
        auto chunk_nbh_it = chunk_nbh_it_in;
        GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) )
        {
          // std::cout<<"dims="<<dims<<" cell_a="<<cell_a<<" loc_a="<<loc_a<<std::endl;
          const double* __restrict__ rx_a = cells[cell_a][field::rx];
          const double* __restrict__ ry_a = cells[cell_a][field::ry];
          const double* __restrict__ rz_a = cells[cell_a][field::rz];
          const size_t n_particles_a = cells[cell_a].size();
          chunk_nbh_it.start_cell( cell_a , n_particles_a );
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            chunk_nbh_it.start_particle( p_a );
            while( ! chunk_nbh_it.end() )
            {
              size_t cell_b=0, p_b=0;
              chunk_nbh_it.get_nbh( cell_b , p_b );
              const Vec3d dr = { cells[cell_b][field::rx][p_b] - rx_a[p_a] , cells[cell_b][field::ry][p_b] - ry_a[p_a] , cells[cell_b][field::rz][p_b] - rz_a[p_a] };
              if( norm2(dr) <= nbh_d2 )
              {
                const size_t pa_idx = cell_particle_offset[cell_a] + p_a;
                const size_t pb_idx = cell_particle_offset[cell_b] + p_b;
                neighbor_list[ particle_nbh_count[pa_idx] ++ ] = pb_idx;
              }
              chunk_nbh_it.next();
            }
          }               
        }
        GRID_OMP_FOR_END
      }
      
      using NeighborOffset = typename FlatPartNbhList::NeighborOffset;
      NeighborOffset latch=0;
      for(size_t i=0;i<total_particles;i++) std::swap( particle_nbh_count[i] , latch );
      assert( latch == particle_nbh_count[total_particles] );
    }
  
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_to_flat_neighbors", make_grid_variant_operator< ChunkNeighbors2FlatParticleLists > );
  }

}

