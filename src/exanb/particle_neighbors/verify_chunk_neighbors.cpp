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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <onika/force_assert.h>

#include <onika/memory/allocator.h>
#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_iterator.h>

namespace exanb
{
  

  template<typename GridT>
  class VerifyChunkNeighbors : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT );
    ADD_SLOT( GridChunkNeighbors , chunk_neighbors , INPUT , REQUIRED );
    ADD_SLOT( GridParticleNeighbors , primary_neighbors , INPUT, REQUIRED );
    ADD_SLOT( GridParticleNeighbors , dual_neighbors , INPUT, REQUIRED );

  public:
    inline void execute () override final
    {    
      size_t cs = chunk_neighbors->m_chunk_size;
      size_t cs_log2 = 0;
      while( cs > 1 )
      {
        ONIKA_FORCE_ASSERT( (cs&1)==0 );
        cs = cs >> 1;
        ++ cs_log2;
      }
      cs = chunk_neighbors->m_chunk_size;
      ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;

      IJK dims = grid->dimension();
      size_t total_nbh = 0;
      size_t total_nbh_chunk = 0;

      auto cells = grid->cells();
      using CellT = std::remove_cv_t< std::remove_reference_t< decltype(cells[0]) > >;

#     pragma omp parallel
      {
        GridParticleNeighborsIterator<2> nbh_it = { { primary_neighbors->data() , dual_neighbors->data() } };
        ChunkParticleNeighborsIterator<CellT,false> chunk_nbh_it = { grid->cells() , chunk_neighbors->m_cell_stream.data() , dims , chunk_neighbors->m_chunk_size };

        GridParticleNeighborsIterator<1> nbh_it_sym = { { dual_neighbors->data() } };
        ChunkParticleNeighborsIterator<CellT,true> chunk_nbh_it_sym = { grid->cells() , chunk_neighbors->m_cell_stream.data() , dims , chunk_neighbors->m_chunk_size };

        GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) reduction(+:total_nbh_chunk,total_nbh) )
        {
          size_t n_particles_a = primary_neighbors->at(cell_a).nbh_start.size();
          ONIKA_FORCE_ASSERT( n_particles_a == dual_neighbors->at(cell_a).nbh_start.size() );

          std::vector< std::set< std::pair<size_t,size_t> > > particle_neighbors( n_particles_a );
          std::vector< std::set< std::pair<size_t,size_t> > > particle_neighbors_sym( n_particles_a );

          // decode compacted chunks
          chunk_nbh_it.start_cell( cell_a , n_particles_a );          
          chunk_nbh_it_sym.start_cell( cell_a , n_particles_a );

          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {            
            ssize_t last_cell = -1;
            ssize_t last_part = -1;
            
            chunk_nbh_it.start_particle( p_a );
            while( ! chunk_nbh_it.end() )
            {
              size_t cell_b=0, p_b=0;
              chunk_nbh_it.get_nbh( cell_b , p_b );
              
              // check monotonicity
              ONIKA_FORCE_ASSERT( static_cast<ssize_t>(cell_b) >= last_cell );
              if( static_cast<ssize_t>(cell_b) != last_cell ) { last_part=-1; }
              last_cell = cell_b;
              ONIKA_FORCE_ASSERT( static_cast<ssize_t>(p_b) >= last_part );
              last_part = p_b;
              
              // insert neighbor cell/particle pair
              particle_neighbors[p_a].insert( std::pair<size_t,size_t>(cell_b,p_b) );
              chunk_nbh_it.next();
            }
            chunk_nbh_it.go_to_next_particle();
            
            chunk_nbh_it_sym.start_particle( p_a );
            while( ! chunk_nbh_it_sym.end() )
            {
              size_t cell_b=0, p_b=0;
              chunk_nbh_it_sym.get_nbh( cell_b , p_b );
              particle_neighbors_sym[p_a].insert( std::pair<size_t,size_t>(cell_b,p_b) );
              chunk_nbh_it_sym.next();
            }
            chunk_nbh_it_sym.go_to_next_particle();
          }          

          // verify that lists from iterator are correct (not non existant particle and no self particle)
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            // no self in neighborhood
            ONIKA_FORCE_ASSERT( particle_neighbors[p_a].find( std::pair<size_t,size_t>(cell_a,p_a) ) == particle_neighbors[p_a].end() );
            ONIKA_FORCE_ASSERT( particle_neighbors_sym[p_a].find( std::pair<size_t,size_t>(cell_a,p_a) ) == particle_neighbors_sym[p_a].end() );

            // no particle index out of bounds
            for( const auto& p : particle_neighbors[p_a] )
            {
              ONIKA_FORCE_ASSERT( p.second < cells[p.first].size() );
            }

            // no particle index out of bounds
            for( const auto& p : particle_neighbors_sym[p_a] )
            {
              ONIKA_FORCE_ASSERT( p.second < cells[p.first].size() );
            }
          }
     
          // verify that chunks contain all neighbors
          nbh_it.start_cell( cell_a , n_particles_a );
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            total_nbh_chunk += particle_neighbors[p_a].size();
            nbh_it.start_particle( p_a );
            while( ! nbh_it.end() )
            {
              ++ total_nbh;
              size_t cell_b=0, p_b=0;
              nbh_it.get_nbh(cell_b, p_b);
              ONIKA_FORCE_ASSERT( particle_neighbors[p_a].find( std::pair<size_t,size_t>(cell_b,p_b) ) != particle_neighbors[p_a].end() );
              nbh_it.next();
            }
          }

          // verify that chunks contain all neighbors (symetric version)
          nbh_it_sym.start_cell( cell_a , n_particles_a );
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            total_nbh_chunk += particle_neighbors[p_a].size();
            nbh_it_sym.start_particle( p_a );
            while( ! nbh_it_sym.end() )
            {
              ++ total_nbh;
              size_t cell_b=0, p_b=0;
              nbh_it_sym.get_nbh(cell_b, p_b);
              ONIKA_FORCE_ASSERT( particle_neighbors[p_a].find( std::pair<size_t,size_t>(cell_b,p_b) ) != particle_neighbors[p_a].end() );
              nbh_it_sym.next();
            }
          }

                    
        }
        GRID_OMP_FOR_END
      }
      
      ldbg << "total_nbh="<<total_nbh<<", total_nbh_chunk="<<total_nbh_chunk<<", ratio="<< total_nbh_chunk/static_cast<double>(total_nbh) << std::endl;
    }
  
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(verify_chunk_neighbors)
  {
   OperatorNodeFactory::instance()->register_factory("verify_chunk_neighbors", make_grid_variant_operator< VerifyChunkNeighbors > );
  }

}

