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

