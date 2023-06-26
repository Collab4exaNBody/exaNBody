#include <exanb/core/operator.h>
#include <exanb/core/domain.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/log.h>
#include <exanb/core/thread.h>

#include <vector>

#include <exanb/grid_cell_particles/move_particles_across_cells.h>

namespace exanb
{
  

  template<class GridT> using MoveAtomsAcrossCells = MovePaticlesAcrossCells<GridT>;
  
   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "move_particles",
      make_grid_variant_operator< MoveAtomsAcrossCells >
      );
  }

}

