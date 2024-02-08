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

#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/simple_particle_neighbors.h>

#include <memory>

namespace exanb
{
  
  struct ComputeSimpleNeighbors : public OperatorNode
  {
    ADD_SLOT(GridParticleNeighbors , primary_neighbors , INPUT , REQUIRED , DocString{"list of neighbors where a < b"} );
    ADD_SLOT(GridParticleNeighbors , dual_neighbors    , INPUT , REQUIRED , DocString{"list of neighbors where a >= b"} );
    ADD_SLOT(GridSimpleParticleNeighbors , simple_neighbors , INPUT_OUTPUT );

    inline void execute () override final
    {
      const GridParticleNeighbors& pb = *primary_neighbors;
      const GridParticleNeighbors& db = *dual_neighbors;

      size_t n_cells = db.size();
      assert( pb.size() == n_cells );
      
      auto& cell_neighbors = simple_neighbors->m_cell_neighbors;
      cell_neighbors.resize( n_cells );

      for(size_t i=0;i<n_cells;i++)
      {
        const unsigned int n_particles = pb[i].nbh_start.size();
        assert( n_particles == db[i].nbh_start.size() );

        cell_neighbors[i].nbh_start.resize( n_particles );
        cell_neighbors[i].neighbors.clear();
        cell_neighbors[i].neighbors.reserve( pb[i].neighbors.size() + db[i].neighbors.size() );
        
        unsigned int nbh_idx_p=0;
        unsigned int nbh_idx_d=0;
        for(unsigned int p=0;p<n_particles;p++)
        {
          for( ; nbh_idx_p < pb[i].nbh_start[p] ; nbh_idx_p++ )
          {
            size_t cell_b=0, p_b=0;
            decode_cell_particle( pb[i].neighbors[nbh_idx_p] , cell_b, p_b );
            cell_neighbors[i].neighbors.push_back( { static_cast<uint32_t>(cell_b) , static_cast<uint32_t>(p_b) } );
          }
          for( ; nbh_idx_d < db[i].nbh_start[p] ; nbh_idx_d++ )
          {
            size_t cell_b=0, p_b=0;
            decode_cell_particle( db[i].neighbors[nbh_idx_d] , cell_b, p_b );
            cell_neighbors[i].neighbors.push_back( { static_cast<uint32_t>(cell_b) , static_cast<uint32_t>(p_b) } );
          }
          cell_neighbors[i].nbh_start[p] = cell_neighbors[i].neighbors.size();
        }
      }

    }
  };

  struct SimpleNeighborsInit : public OperatorNode
  {
    ADD_SLOT(GridSimpleParticleNeighbors , chunk_neighbors , INPUT_OUTPUT );
    inline void execute () override final
    {
      *chunk_neighbors = GridSimpleParticleNeighbors{};
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "compute_simple_neighbors" , make_simple_operator< ComputeSimpleNeighbors > );
   OperatorNodeFactory::instance()->register_factory( "simple_neighbors_init" , make_simple_operator< SimpleNeighborsInit > );
  }

}

