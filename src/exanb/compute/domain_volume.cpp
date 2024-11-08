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
#include <exanb/core/domain.h>

namespace exanb
{
  
  struct DomainVolumeNode : public OperatorNode
  {  
    ADD_SLOT( Domain  , domain, INPUT, REQUIRED  , DocString{"Simulation domain"});
    ADD_SLOT( double  , volume, INPUT_OUTPUT      , DocString{"Computed volume"}  );
    
  public:
    
    inline void execute () override final
    {
      *volume = 1.0;
      if( ! domain->xform_is_identity() )
        {
          Mat3d mat = domain->xform();
          Vec3d a { mat.m11, mat.m21, mat.m31 };
          Vec3d b { mat.m12, mat.m22, mat.m32 };
          Vec3d c { mat.m13, mat.m23, mat.m33 };
          *volume = dot( cross(a,b) , c );
        }
      *volume *= bounds_volume( domain->bounds() );
      ldbg << "Domain volume = " << *volume << std::endl;
    }

    inline std::string documentation() const override final
    {
      return "This operator computes the scalar volume of the domain with any shape.";
    }
  };
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "domain_volume", make_simple_operator< DomainVolumeNode > );
  }

}

