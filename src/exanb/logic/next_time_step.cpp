#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/log.h>

#include <memory>

namespace exanb
{

  struct NextTimeStepNode : public OperatorNode
  {  
    ADD_SLOT( long   , timestep      , INPUT_OUTPUT );
    ADD_SLOT( double , dt            , INPUT        , REQUIRED );
    ADD_SLOT( double , physical_time , INPUT_OUTPUT );
    // this is equivalent to
    // OperatorSlot<int64_t> timestep { this, "timestep", INPUT_OUTPUT };

    inline void execute() override final
    {
      ++ *timestep;
      *physical_time += *dt;
      //ldbg << "timstep       -> " << timestep <<std::endl;
      //std::cout << "physical_time -> " << *physical_time <<std::endl;
    }

  };
 
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "next_time_step", make_compatible_operator< NextTimeStepNode > );
  }

}

