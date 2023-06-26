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
	  ONIKA_CU_MEM_PREFETCH( cells[i].storage_ptr() , cells[i].storage_size() , d , gpu_execution_context()->m_cuda_ctx->m_threadStream[0] );
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

