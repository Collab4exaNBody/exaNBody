#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <memory>

namespace exanb
{

  class ValueScaleOperator : public OperatorNode
  {
    ADD_SLOT( double , scale  , INPUT , 1.0  );  // scale factor
    ADD_SLOT( double , in_value  , INPUT  );  // value to scale
    ADD_SLOT( double , out_value  , INPUT  );  // result

  public:
    inline void execute () override final
    {
      *out_value = (*in_value) * (*scale);
    }
    
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("value_scale" , make_simple_operator< ValueScaleOperator > );
  }

}

