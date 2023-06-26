#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

#include <memory>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class BooleanSetNode : public OperatorNode
  {
  public:
  
    ADD_SLOT( bool , value , INPUT_OUTPUT);
    ADD_SLOT( bool , set , INPUT);
    
    void execute() override final
    {
      *value = *set;
    }

  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "boolean_set", make_compatible_operator< BooleanSetNode > );
  }

}

