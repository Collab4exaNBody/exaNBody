#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

#include <memory>

namespace exanb
{

  class BooleanOrNode : public OperatorNode
  {
  public:
  
    ADD_SLOT( bool , in1 , INPUT , REQUIRED );
    ADD_SLOT( bool , in2 , INPUT , REQUIRED );
    ADD_SLOT( bool , result , OUTPUT );
    
    void execute() override final
    {
      *result = *in1 || *in2;
    }

  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "boolean_or", make_compatible_operator< BooleanOrNode > );
  }

}

