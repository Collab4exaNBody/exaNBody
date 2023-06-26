#include <exanb/core/thread.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <memory>

namespace exanb
{

  template<class GridT>
  struct ResizeParticleLocksNode : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT);
    ADD_SLOT( GridParticleLocks , particle_locks ,INPUT_OUTPUT);

    inline void execute ()  override final
    {      
      IJK dims = grid->dimension();
      size_t n_cells = grid->number_of_cells();
      ldbg << "resize_particle_locks: cells: "<<particle_locks->size() <<" -> " << n_cells << std::endl;
      
      particle_locks->resize( n_cells );
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc)
        {
          particle_locks->at(i).resize( grid->cell_number_of_particles(i) );
        }
        GRID_OMP_FOR_END
      }

    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "resize_particle_locks", make_grid_variant_operator< ResizeParticleLocksNode > );
  }

}

