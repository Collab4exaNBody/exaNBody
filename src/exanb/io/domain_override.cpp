
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct DomainOverride : public OperatorNode
  {
    ADD_SLOT( Domain , domain , INPUT_OUTPUT );
    ADD_SLOT( double , cell_size , INPUT , OPTIONAL );
    ADD_SLOT( AABB , bounds , INPUT , OPTIONAL );
    ADD_SLOT( IJK , grid_dims , INPUT , OPTIONAL );
    ADD_SLOT( bool , expandable , INPUT , OPTIONAL );
    ADD_SLOT( std::vector<bool> , periodic , INPUT , OPTIONAL );

    inline void execute() override final
    {
      if( cell_size.has_value() )
      {
        lout << "override domain's cell size to "<< *cell_size << std::endl;
        domain->set_cell_size( *cell_size );
      }
      
      if( bounds.has_value() )
      {
        lout << "override domain's bounds to "<< *bounds << std::endl;
        domain->set_bounds( *bounds );
      }

      if( grid_dims.has_value() )
      {
        lout << "override domain's grid_dims to "<< *grid_dims << std::endl;
        domain->set_grid_dimension( *grid_dims );
      }

      if( periodic.has_value() )
      {
        auto p = *periodic;
        p.resize(3,false);
        lout << "override domain's periodicity to "<<p[0]<<" "<<p[1]<<" "<<p[2]<< std::endl;
        domain->set_periodic_boundary( p[0], p[1], p[2] );
      }

      if( expandable.has_value() )
      {
        lout << "override domain's expandable to "<< *expandable << std::endl;
        domain->set_expandable( *expandable );
      }

      ldbg<<"domain override: "<< *domain << std::endl;
      //check_domain( *domain );
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "domain_override", make_simple_operator<DomainOverride> );
  }

}

