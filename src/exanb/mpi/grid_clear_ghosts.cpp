#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>

namespace exanb
{
  

  template<class GridT>
  class GridClearGhosts : public OperatorNode
  {  
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( GridT              , grid      , INPUT_OUTPUT );

  public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return "Empties ghost cells";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      auto & g = *grid;
      apply_grid_shell( grid->dimension() , 0 , grid->ghost_layers() , [&g](ssize_t i, const IJK&){ g.cell(i).clear( g.cell_allocator() ); } );
#     ifndef NDEBUG
      const size_t n_cells = grid->number_of_cells();
      for(size_t i=0;i<n_cells;i++)
      {
        if( g.is_ghost_cell(i) ) { assert( g.cell(i).empty() && g.cell(i).capacity()==0 && g.cell(i).storage_ptr()==nullptr ); }
      }
#     endif
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "grid_clear_ghosts", make_grid_variant_operator< GridClearGhosts > );
  }

}

