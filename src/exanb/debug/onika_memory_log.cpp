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

