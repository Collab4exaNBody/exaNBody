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

#include <memory>

namespace exanb
{

  class ValueScaleOperator : public OperatorNode
  {
    ADD_SLOT( double , scale  , INPUT , 1.0  );  // scale factor
    ADD_SLOT( double , in_value  , INPUT  );  // value to scale
    ADD_SLOT( double , out_value  , INPUT  );  // result

  public:
    inline void execute () override final
    {
      *out_value = (*in_value) * (*scale);
    }
    
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(value_scale_operator)
  {
   OperatorNodeFactory::instance()->register_factory("value_scale" , make_simple_operator< ValueScaleOperator > );
  }

}

