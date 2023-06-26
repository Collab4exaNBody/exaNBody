#include <exanb/mpi/simple_cost_model.h>
#include <exanb/core/make_grid_variant_operator.h>

namespace exanb
{

  template<class GridT> using SimpleCostModelTmpl = SimpleCostModel<GridT>;

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "simple_cost_model",
      make_grid_variant_operator< SimpleCostModelTmpl > );
  }

}

