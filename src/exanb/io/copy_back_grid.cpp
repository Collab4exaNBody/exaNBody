#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>

namespace exanb
{

  template<typename GridT>
  struct CopyBackGrid : public OperatorNode
  {
    ADD_SLOT(GridT       , grid       , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT(GridT       , grid_copy  , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT(bool        , clear_copy , INPUT , false );

    inline void execute () override final
    {
      *grid = *grid_copy;
      if ( *clear_copy ) { grid_copy->reset(); }
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "copy_back_grid", make_grid_variant_operator< CopyBackGrid > );
  }

}
