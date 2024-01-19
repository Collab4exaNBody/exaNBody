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
// #pragma xstamp_cuda_enable  // DO NOT REMOVE THIS LINE !!

#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>

#include <onika/cuda/cuda_context.h>

namespace exanb
{
  

  template<class GridT>
  struct GridGPUPrefetch : public OperatorNode
  {
    ADD_SLOT(GridT , grid , INPUT_OUTPUT , DocString{"Particle grid"} );
    ADD_SLOT(bool  , ghost               , INPUT , false );

    inline void execute () override final
    {
      // const bool compact_ghosts = *ghost;
      auto cells = grid->cells();
      IJK dims = grid->dimension();
      ssize_t gl = (*ghost) ? 0 : grid->ghost_layers() ;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims-2*gl,_,loc, schedule(dynamic) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gl );
	  ONIKA_CU_MEM_PREFETCH( cells[i].storage_ptr() , cells[i].storage_size() , 0 , parallel_execution_context()->m_cuda_ctx->m_threadStream[0] );
	}
	GRID_OMP_FOR_END
      }
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "grid_gpu_prefetch", make_grid_variant_operator<GridGPUPrefetch> );
  }

}

