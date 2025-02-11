#include <md/potential/snap/snap_force.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/cpp_utils.h>

namespace md
{

  template<class GridT> using SnapNewForceTmpl = SnapNewForce<GridT>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT(snap_force)
  {
    OperatorNodeFactory::instance()->register_factory( "snap_force" ,make_grid_variant_operator< SnapNewForceTmpl > );
  }

}


