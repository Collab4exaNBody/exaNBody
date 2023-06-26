#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct DomainSetXForm : public OperatorNode
  {
  
    ADD_SLOT( Mat3d , xform , INPUT , REQUIRED );
    ADD_SLOT( Domain , domain , INPUT_OUTPUT );

    inline void execute ()  override final
    {
      domain->set_xform( *xform );
    }

  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "domain_set_xform", make_compatible_operator< DomainSetXForm > );
  }

}


