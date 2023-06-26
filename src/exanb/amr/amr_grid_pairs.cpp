#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct AmrGridSubCellPairs : public OperatorNode
  {
    ADD_SLOT( AmrGrid , amr            , INPUT , REQUIRED );
    ADD_SLOT( double  , nbh_dist       , INPUT , REQUIRED );  // value added to the search distance to update neighbor list less frequently
    ADD_SLOT( Domain  , domain         , INPUT_OUTPUT );
    ADD_SLOT( AmrSubCellPairCache , amr_grid_pairs , INPUT_OUTPUT );

    inline void execute () override final
    {
      max_distance_sub_cell_pairs( ldbg , *amr , domain->cell_size() , *nbh_dist , *amr_grid_pairs );
    }

  private:  
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("amr_grid_pairs", make_simple_operator< AmrGridSubCellPairs > );
  }

}

