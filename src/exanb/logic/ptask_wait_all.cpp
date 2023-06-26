#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

namespace exanb
{

  struct ParallelTaskQueueWaitAll : public OperatorNode
  {
    ADD_SLOT( bool , synchronize , INPUT , false );

    inline void generate_tasks () override final
    {
      ptask_queue().wait_all();
    }

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "ptask_wait_all", make_compatible_operator<ParallelTaskQueueWaitAll> );
  }

}

