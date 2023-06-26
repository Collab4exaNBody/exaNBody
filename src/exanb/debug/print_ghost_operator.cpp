#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/core/print_particle.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <algorithm>
#include <sstream>

namespace exanb
{
  
  
  // =================== utility functions ==========================
  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_id>
    >
  class PrintGhosts : public OperatorNode
  {
    using ParticleIds = std::vector<uint64_t>;
    ADD_SLOT( GridT                    , grid                  , INPUT );

  public:
    inline void execute () override final
    {
      GridT& grid = *(this->grid);

      auto cells = grid.cells();
      size_t n_cells = grid.number_of_cells();

      lout << "DIMENSION : " << grid.dimension() << std::endl;
      lout << "*******************************GHOSTS****************************************" << std::endl;

#     pragma omp parallel
      {
#       pragma omp for
        for(size_t i=0;i<n_cells;i++)
        {
          if( grid.is_ghost_cell(i) )
            {
              size_t n_part = cells[i].size();
              for(size_t j=0;j<n_part;j++)
                {
#                 pragma omp critical
                  {
                    //lout << "offset : " << cell_pos.i << " " << cell_pos.j << " " << cell_pos.i << std::endl;
                    print_particle( lout , cells[i][j] );
                  }
                }
            }
        }
      }

      lout << "*******************************REAL PARTICLES****************************************" << std::endl;


#     pragma omp parallel
      {
#       pragma omp for
        for(size_t i=0;i<n_cells;i++)
          {
            //Check if cell is a ghost cell
            if(!grid.is_ghost_cell(i) )
              {
                size_t n_part = cells[i].size();
                for(size_t j=0;j<n_part;j++)
                  {
#                 pragma omp critical
                    {
                      //lout << "offset : " << cell_pos.i << " " << cell_pos.j << " " << cell_pos.i << std::endl;
                      print_particle( lout , cells[i][j] );
                    }
                  }
              }
          }
      }

    }

  };


  template<class GridT> using PrintGhostsTmpl = PrintGhosts<GridT>;
    

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "debug_print_ghosts", make_grid_variant_operator< PrintGhostsTmpl > );
  }

}
