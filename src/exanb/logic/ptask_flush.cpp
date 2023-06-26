#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

namespace exanb
{

  struct ParallelTaskQueueFlush : public OperatorNode
  {
    ADD_SLOT( bool , synchronize , INPUT , false );

    inline void generate_tasks () override final
    {
      ptask_queue().flush();
    }

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "ptask_flush", make_compatible_operator<ParallelTaskQueueFlush> );
  }

}

