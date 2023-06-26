#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/check_particles_inside_cell.h>

namespace exanb
{
  
  
  template<typename GridT>
  struct CheckParticlesInsideCells : public OperatorNode
  { 
    ADD_SLOT( GridT , grid , INPUT );
    
    inline void execute () override final
    {
      if( ! check_particles_inside_cell(*grid) )
      {
        std::abort();
      }
    }
        
  };
  
   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "check_particles_inside_cells",
      make_grid_variant_operator< CheckParticlesInsideCells >
      );
  }

}

