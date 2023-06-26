#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

namespace exanb
{

  template< class GridT >
  class GridPassThru : public OperatorNode
  {
    ADD_SLOT( GridT          , grid    , INPUT_OUTPUT);
  public:
    inline void execute ()  override final {}
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "grid_passthru", make_grid_variant_operator<GridPassThru> );
  }

}

