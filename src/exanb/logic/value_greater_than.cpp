#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class ValueGreaterThanNode : public OperatorNode
  {  
    ADD_SLOT( double , value     , INPUT , REQUIRED );
    ADD_SLOT( double , threshold , INPUT , REQUIRED);
    ADD_SLOT( bool   , result    , OUTPUT);
  public: 
    void execute() override final
    {
      *result = *value > *threshold;
      ldbg << "ValueGreaterThanNode: value="<<(*value)<<", threshold="<<(*threshold)<<", result="<<std::boolalpha<<(*result)<<std::endl;
    }
  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "greater_than", make_compatible_operator< ValueGreaterThanNode > );
  }

}

