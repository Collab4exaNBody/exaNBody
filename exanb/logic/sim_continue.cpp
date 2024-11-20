/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/yaml/yaml_utils.h>
#include <memory>

#include <signal.h>

#ifdef __use_lib_ccc_user
#include "ccc_user.h"
#endif

namespace exanb
{

  static bool s_simulation_termination_requested = false;
  static inline void simulation_termination_handler (int sig, siginfo_t *info, void *ucontext)
  {
    if( sig == SIGINT )
    {
      if( s_simulation_termination_requested )
      {
        lout << "\nTermination forced, aborting ..." << std::endl << std::flush;
        std::abort();
      }
      else
      {
        lout << "\nTermination requested, please wait ..." << std::endl << std::flush;
        s_simulation_termination_requested = true;
      }
    }
  }

  class SimContinueOperator : public OperatorNode
  {
    struct TerminationSignalHandler
    {
      bool initialized = false;
    };

  public:
  
    ADD_SLOT( long , timestep , INPUT, REQUIRED);
    ADD_SLOT( long , end_at , INPUT , REQUIRED);
    ADD_SLOT( std::string , stop_file , INPUT , std::string("stop") );
    ADD_SLOT( double , remain , INPUT , 600.0 );
    ADD_SLOT( long , check_freq , INPUT , 10 );
    ADD_SLOT( bool , result , INPUT_OUTPUT);
    
    ADD_SLOT( TerminationSignalHandler, signal_handler_priv , PRIVATE );
    
    void execute() override final
    {
      double tremain = 999999.99;
      bool stop_request = false;
      bool checked = false;

      if( ! signal_handler_priv->initialized )
      {
        struct sigaction act;
        act.sa_handler = nullptr;
        act.sa_sigaction = simulation_termination_handler;
        sigemptyset( & act.sa_mask );
        act.sa_flags = SA_SIGINFO;
        sigaction (SIGINT, & act, nullptr);
        signal_handler_priv->initialized = true;
      }

      if( (*timestep) % (*check_freq) == 0 )
      {
        checked = true;
#       ifdef __use_lib_ccc_user
//#warning USING ccc_user
        ccc_tremain(&tremain);
#       endif
        stop_request = std::ifstream( *stop_file ).good();
      }

      *result = ( *timestep <= *end_at ) && ( tremain > *remain ) && !stop_request && !s_simulation_termination_requested;

      if( checked )
      {
        ldbg << "SimContinue: timestep=" <<(*timestep)
           <<", tremain="<<tremain
           <<", stop="<<stop_request
           <<", remain="<<(*remain)
           <<", result="<<std::boolalpha<< (*result) <<std::endl;
      }
    }
 
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(sim_continue)
  {
    OperatorNodeFactory::instance()->register_factory( "sim_continue", make_simple_operator<SimContinueOperator> );
  }

}

