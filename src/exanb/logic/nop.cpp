#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>

namespace exanb
{

  struct NoOperationNode : public OperatorNode
  {  
    inline NoOperationNode() { set_profiling(false); }
    inline void execute() override final {}
  };
 
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "nop", make_compatible_operator< NoOperationNode > );
  }

}

