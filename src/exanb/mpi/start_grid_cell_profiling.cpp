#include <exanb/core/grid.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <cmath>

namespace exanb
{

  // simple cost model where the cost of a cell is the number of particles in it
  // 
  template< class GridT >
  class StartGridCellProfiling : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT_OUTPUT , REQUIRED);
    
  public:

    inline void execute () override final
    {
      grid->set_cell_profiling( true );
      grid->reset_cell_profiling_data();
      //std::cout<<"cell profiling = "<<grid->cell_profiling();
    }

  };
  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "start_grid_cell_profiling",
      make_grid_variant_operator< StartGridCellProfiling > );
  }

}

