#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <onika/memory/allocator.h>

namespace exanb
{

  struct OnikaMemoryLog : public OperatorNode
  {  
    ADD_SLOT( bool , enable , INPUT , true , DocString{"activate/deactivate onika memory allocator debug logs"} );

    inline void execute () override final
    {
      onika::memory::GenericHostAllocator::set_debug_log(*enable);
    }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "onika_memory_log", make_simple_operator<OnikaMemoryLog> );
  }

}

