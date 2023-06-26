#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/print_utils.h>
#include <exanb/core/print_particle.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <map>
#include <algorithm>
#include <sstream>

namespace exanb
{

  // ================== Thermodynamic state compute operator ======================

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_id >
    >
  class DebugParticleNode : public OperatorNode
  {
    using ParticleIds = std::vector<uint64_t>;

    ADD_SLOT( GridT       , grid      , INPUT, REQUIRED);
    ADD_SLOT( ParticleIds , ids       , INPUT, REQUIRED);
    ADD_SLOT( bool        , ghost     , INPUT, false );

  public:
  
    inline void execute () override final
    {
      ParticleIds ids = *(this->ids);
      std::sort( ids.begin(), ids.end() );

      auto cells = grid->cells();
      IJK dims = grid->dimension();
      
      std::map<uint64_t,std::vector<std::string> > dbg_items;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims, i, loc )
        {
          const uint64_t* __restrict__ part_ids = cells[i][field::id];
          bool is_ghost_cell = grid->is_ghost_cell( loc );
          size_t n_part = cells[i].size();
          for(size_t j=0;j<n_part;j++)
          {
            if( ( ids.empty() || std::binary_search( ids.begin(), ids.end(), part_ids[j] ) ) && ( (*ghost) || !is_ghost_cell ) )
            {
              std::ostringstream oss;
              oss<< default_stream_format;
              oss<<"---- PARTICLE "<<part_ids[j]<<" ";
              if(is_ghost_cell) { oss<<"GHOST"; }
              oss<<"----"<<std::endl<<"cell = " << loc <<std::endl;
              print_particle( oss , cells[i][j] );
              oss<<"------------------------------------------";
#             pragma omp critical
              {
                dbg_items[ part_ids[j] ].push_back( oss.str() );
              }
            }
          }
        }
        GRID_OMP_FOR_END
      }

      for( const auto& x : dbg_items ) for( const auto& y : x.second )
      {
        lout << y << std::endl;
      }
      
    }

  };
  
  template<class GridT> using DebugParticleNodeTmpl = DebugParticleNode<GridT>;
  
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "debug_particle", make_grid_variant_operator<DebugParticleNodeTmpl> );
  }

}

