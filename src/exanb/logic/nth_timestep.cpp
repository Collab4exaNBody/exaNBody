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

  struct NthTimeStepCache
  {
    int64_t m_last_time_step = 0;
    bool m_first_time = true;
  };

  class NthTimeStepNode : public OperatorNode
  {
    ADD_SLOT( long , timestep , INPUT);

    ADD_SLOT( bool , first , INPUT, true );
    ADD_SLOT( long , freq , INPUT , 1 );
    ADD_SLOT( bool , delayed , INPUT , false );
    ADD_SLOT( long , at_timestep , INPUT , -1 ); // exact timestep match

    ADD_SLOT( bool , result , INPUT_OUTPUT);
    ADD_SLOT( NthTimeStepCache , nth_timestep_cache , PRIVATE);
    
  public:
    void execute() override final
    {
      if( *at_timestep != -1 )
      {
        *result = ( (*at_timestep) == (*timestep) );
      }
      else if( nth_timestep_cache->m_first_time )
      {
        *result = (*first);
        nth_timestep_cache->m_first_time = false;
      }
      else if( (*freq) > 0 )
      {
        *result = 
             ( ( (*timestep) % (*freq) ) == 0 )
          || ( (*delayed) && ( ( (*timestep) - (*freq) ) >= nth_timestep_cache->m_last_time_step ) );
      }
      else
      {
        *result = false;
      }
      
      if( *result )
      {
        nth_timestep_cache->m_last_time_step = (*timestep);
      }
      
      ldbg << "NthTimeStepNode: timestep="<< *timestep <<", freq="<< *freq<<", delayed="<<std::boolalpha<< *delayed<< ", result="<<std::boolalpha<< *result <<std::endl;
    }

  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "nth_timestep", make_compatible_operator< NthTimeStepNode > );
  }

}

