#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>

namespace exanb
{

  template<class GridT>
  struct GridClear : public OperatorNode
  {
    ADD_SLOT(GridT , grid , INPUT_OUTPUT , DocString{"Particle grid"} );

    inline bool is_sink() const override final { return true; } // not a suppressable operator

    inline void execute () override final
    {
      auto cells = grid->cells();
      size_t n_cells = grid->number_of_cells();
#     pragma omp parallel
      {     
#       pragma omp for schedule(dynamic)
        for(size_t i=0;i<n_cells;i++)
        {
          cells[i].clear();
        }
      }
      grid->reset();
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "grid_clear", make_grid_variant_operator<GridClear> );
  }

}

