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
#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>

#include <onika/cuda/cuda_context.h>
#include <onika/memory/allocator.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>

#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_scratch.h>
#include <exanb/particle_neighbors/chunk_neighbors_host_write_accessor.h>

#include <exanb/core/domain.h>
#include <exanb/core/xform.h>

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

namespace exanb
{

  template<typename GridT>
  struct BuildChunkNeighbors : public OperatorNode
  {
    ADD_SLOT( GridT               , grid            , INPUT );
    ADD_SLOT( AmrGrid             , amr             , INPUT );
    ADD_SLOT( AmrSubCellPairCache , amr_grid_pairs  , INPUT );
    ADD_SLOT( Domain              , domain          , INPUT );
    ADD_SLOT( double              , nbh_dist_lab    , INPUT );
    ADD_SLOT( GridChunkNeighbors  , chunk_neighbors , INPUT_OUTPUT );

    ADD_SLOT( ChunkNeighborsConfig, config , INPUT_OUTPUT, ChunkNeighborsConfig{} );
    ADD_SLOT( ChunkNeighborsScratchStorage, chunk_neighbors_scratch, PRIVATE );

    inline void execute () override final
    {
      unsigned int cs = config->chunk_size;
      unsigned int cs_log2 = 0;
      while( cs > 1 )
      {
        assert( (cs&1)==0 );
        cs = cs >> 1;
        ++ cs_log2;
      }
      cs = 1<<cs_log2;
      ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;
      if( cs != static_cast<size_t>(config->chunk_size) )
      {
        lerr<<"chunk_size is not a power of two"<<std::endl;
        std::abort();
      }

      const bool gpu_enabled = (global_cuda_ctx() != nullptr) ? global_cuda_ctx()->has_devices() : false;
      static constexpr std::false_type no_z_order = {};

      if( config->half_symetric || config->skip_ghosts )
      {
        if( cs != 1 )
        {
          lerr<<"ChunkNeighbors: WARNING: half_symetric="<<std::boolalpha<<config->half_symetric<<", skip_ghosts="<<std::boolalpha<<config->skip_ghosts<<", so chunk_size is forced to 1" << std::endl;
          config->chunk_size = 1;
          cs = 1;
          cs_log2 = 0;
        }
      }

      LinearXForm xform = { domain->xform() };
      NeighborFilterHalfSymGhost<GridT> nbh_filter = { *grid , config->half_symetric , config->skip_ghosts };
      chunk_neighbors_execute(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter );
    }

  };

  struct ChunkNeighborsInit : public OperatorNode
  {
    ADD_SLOT(GridChunkNeighbors   , chunk_neighbors , INPUT_OUTPUT );
    inline void execute () override final
    {
      *chunk_neighbors = GridChunkNeighbors{};
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors", make_grid_variant_operator< BuildChunkNeighbors > );
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_init", make_simple_operator< ChunkNeighborsInit > );
  }

}

