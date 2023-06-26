#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

namespace exanb
{

  struct LoadBalanceEventCounterOperator : public OperatorNode
  {  
    ADD_SLOT( bool   , lb_flag             , INPUT_OUTPUT, false );
    ADD_SLOT( bool   , move_flag           , INPUT_OUTPUT, false );
    ADD_SLOT( bool   , domain_extended     , INPUT_OUTPUT, false );
    ADD_SLOT( double , lb_inbalance        , INPUT_OUTPUT, 0.0 );

    ADD_SLOT( long   , lb_counter          , INPUT_OUTPUT, 0 );
    ADD_SLOT( long   , move_counter        , INPUT_OUTPUT, 0 );
    ADD_SLOT( long   , domain_ext_counter  , INPUT_OUTPUT, 0 );
    ADD_SLOT( double , lb_inbalance_max    , INPUT_OUTPUT, 0.0 );

    inline void execute () override final
    {      
      if( *lb_flag )         { ++ (*lb_counter); }
      if( *move_flag )       { ++ (*move_counter); }
      if( *domain_extended ) { ++ (*domain_ext_counter); }
      *lb_inbalance_max = std::max( *lb_inbalance_max , *lb_inbalance );
      //lout << "lb="<< *lb_flag << ", mv="<<*move_flag<< ", de="<<*domain_extended << ", lbc="<<*lb_counter<<", mvc="<<*move_counter<<", dec="<<*domain_ext_counter<<std::endl;
      *lb_flag = false;
      *move_flag = false;
      *domain_extended = false;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "lb_event_counter", make_simple_operator<LoadBalanceEventCounterOperator> );
  }

}

