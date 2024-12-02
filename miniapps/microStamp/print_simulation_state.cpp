#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/domain.h>

namespace microStamp
{
  using namespace exanb;
  using SimulationState = std::vector<double>;

  class PrintSimulationState : public OperatorNode
  {  
    // thermodynamic state & physics data
    ADD_SLOT( long               , timestep            , INPUT , REQUIRED );
    ADD_SLOT( double             , physical_time       , INPUT , REQUIRED );
    ADD_SLOT( SimulationState , simulation_state , INPUT , REQUIRED );

    // LB and particle movement statistics
    ADD_SLOT( long               , lb_counter          , INPUT_OUTPUT );
    ADD_SLOT( long               , move_counter        , INPUT_OUTPUT );
    ADD_SLOT( long               , domain_ext_counter  , INPUT_OUTPUT );
    ADD_SLOT( double             , lb_inbalance_max    , INPUT_OUTPUT );

    // NEW
    ADD_SLOT(Domain              , domain              , INPUT , OPTIONAL, DocString{"Deformation box matrix"} );

  public:
    inline bool is_sink() const override final { return true; }
  
    inline void execute () override final
    {      
      lout << "T=" << (*physical_time) << " , N="<< simulation_state->at(2) << " , Kin.E="<<simulation_state->at(0)<< " , Pot.E="<<simulation_state->at(1) << std::endl;
    }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "print_simulation_state", make_simple_operator<PrintSimulationState> );
  }

}

