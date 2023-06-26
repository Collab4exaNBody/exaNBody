#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>

namespace exanb
{

  template<typename GridT>
  struct CopyGridperator : public OperatorNode
  {
    ADD_SLOT(GridT       , grid       , INPUT , REQUIRED );
    ADD_SLOT(GridT       , grid_copy  , OUTPUT );

    inline void execute () override final
    {
      *grid_copy = *grid;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "copy_grid", make_grid_variant_operator< CopyGridperator > );
  }

}
