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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/defbox/deformation.h>
#include <exanb/defbox/deformation_yaml.h>
#include <exanb/defbox/deformation_stream.h>
#include <exanb/defbox/deformation_math.h>
#include <exanb/core/domain.h>

#include <exanb/core/text_enum.h>

XSTAMP_TEXT_ENUM_START(XFormUpdateMode);
XSTAMP_TEXT_ENUM_ITEM(SET);
XSTAMP_TEXT_ENUM_ITEM(PREMULTIPLY);
XSTAMP_TEXT_ENUM_ITEM(POSTMULTIPLY);
XSTAMP_TEXT_ENUM_ITEM(IGNORE);
XSTAMP_TEXT_ENUM_END();

XSTAMP_TEXT_ENUM_YAML(XFormUpdateMode);


namespace exanb
{

  struct DeformationXFormNode : public OperatorNode
  {
    using XFormVec = std::vector<Mat3d>;
    using TimeVec = std::vector<double>;

    ADD_SLOT( Deformation , defbox , INPUT , REQUIRED );
    ADD_SLOT( XFormUpdateMode      , update_mode , INPUT, XFormUpdateMode::SET );
    ADD_SLOT( Domain     , domain  , INPUT_OUTPUT );
    ADD_SLOT( Mat3d      , xform   , OUTPUT ); // outputs deformation matrix ( not combined with domain's matrix)
    ADD_SLOT( Mat3d      , inv_xform , OUTPUT ); // outputs inverse of deformation matrix

    inline void execute () override final
    {   
      assert( defbox_check_angles( defbox->m_angles ) );
      
      if( defbox->m_angles==Vec3d{M_PI/2,M_PI/2,M_PI/2} && defbox->m_extension==Vec3d{1.,1.,1.} )
      {
        *xform = make_identity_matrix();
      }
      else
      {
        *xform = deformation_to_matrix( *defbox );
      }
      *inv_xform = inverse( *xform );
  
      //std::cout << update_mode->str() << std::endl;

      if( *update_mode == XFormUpdateMode::SET ) { domain->set_xform( *xform ); }
      else if( *update_mode == XFormUpdateMode::PREMULTIPLY ) { domain->set_xform( (*xform) * domain->xform() ); }
      else if( *update_mode == XFormUpdateMode::POSTMULTIPLY ) { domain->set_xform( domain->xform() * (*xform) ); }

      ldbg << "defbox = " << *defbox << std::endl;
      if( *update_mode != XFormUpdateMode::IGNORE )
      {
        ldbg << *domain << std::endl;
      }
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "deformation_xform", make_compatible_operator< DeformationXFormNode > );
  }

}


