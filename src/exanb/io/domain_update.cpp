
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct DomainUpdateNode : public OperatorNode
  {
    ADD_SLOT( Domain , domain , INPUT_OUTPUT );

    inline void execute() override final
    {
      compute_domain_bounds( *domain, ReadBoundsSelectionMode::DOMAIN_BOUNDS );
      ldbg<<"domain update: "<< *domain << std::endl;
      check_domain( *domain );
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "domain_update", make_simple_operator<DomainUpdateNode> );
  }

}

