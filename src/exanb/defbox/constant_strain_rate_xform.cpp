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
#include <onika/math/basic_types_operators.h>
//#include "exanb/container_utils.h"
#include <onika/string_utils.h>
#include <exanb/core/domain.h>
#include <onika/physics/units.h>
#include <string>
#include <math.h>


using namespace std;

namespace exanb
{

  struct ConstantStrainRateXFormNode : public OperatorNode
  {

    ADD_SLOT( std::string  , mode         , INPUT , REQUIRED );
    ADD_SLOT( double       , strain_rate  , INPUT , REQUIRED );
    ADD_SLOT( Vec3d        , direction1   , INPUT , Vec3d{1.0, 0.0, 0.0} );
    ADD_SLOT( Vec3d        , direction2   , INPUT , Vec3d{0.0, 1.0, 0.0} );
    ADD_SLOT( Vec3d        , direction3   , INPUT , Vec3d{0.0, 0.0, 1.0} );
    ADD_SLOT( Vec3d        , diagpredef    , INPUT , Vec3d{1.0, 1.0, 1.0} );    
    ADD_SLOT( double       , sign1        , INPUT , 1.0 );
    ADD_SLOT( double       , sign2        , INPUT , 1.0 );
    ADD_SLOT( double       , sign3        , INPUT , 1.0 );
    ADD_SLOT( double       , dt           , INPUT , REQUIRED );
    ADD_SLOT( long         , timestep     , INPUT , REQUIRED );
    ADD_SLOT( double       , time_start   , INPUT , 0.0 );    
    ADD_SLOT( Domain       , domain       , INPUT_OUTPUT );
    ADD_SLOT( Mat3d        , macroscopic_strain       , OUTPUT );
    
    inline void execute ()  override final
    {

      double dt_sec = EXANB_QUANTITY( (*dt) / s ); //UnityConverterHelper::convert(*dt, "1/s");
      double prectime = dt_sec * (*timestep - 1);      
      double curtime = dt_sec * (*timestep);
      double starttime = EXANB_QUANTITY( (*time_start) / s ); //UnityConverterHelper::convert(*time_start, "1/s");
      Mat3d Id33 = make_identity_matrix();
      std::string mode = *(this->mode);
      Mat3d xform = domain->xform();
      Mat3d xformlocal = Id33;      
      Mat3d predef = diag_matrix(*diagpredef);
      Mat3d macroscopic_strain = make_identity_matrix();
      Mat3d F = Id33;
      Mat3d Fprec = Id33;

      if(mode == "uniaxial"){
	Vec3d m = *direction1;
	Mat3d m_x_m = tensor( m, m );
	double factm = (*sign1) * (*strain_rate) * (curtime - starttime);
	double factmprec = (*sign1) * (*strain_rate) * (prectime - starttime);	
	if(curtime < starttime){
	  F = Id33;
	  Fprec = Id33;	  
	  xformlocal = F * inverse(Fprec) * predef;
	} else {
	  F = Id33 + m_x_m * factm;
	  Fprec = Id33 + m_x_m * factmprec;
	  xformlocal = F * inverse(Fprec) * predef;	  
	}
      } else if(mode == "biaxial" ){
	Vec3d m = *direction1;
	Vec3d n = *direction2;	
	Mat3d m_x_m = tensor( m, m );
	Mat3d n_x_n = tensor( n, n );	
	double factm = (*sign1) * (*strain_rate) * (curtime - starttime);
	double factn = (*sign2) * (*strain_rate) * (curtime - starttime);
	double factmprec = (*sign1) * (*strain_rate) * (prectime - starttime);
	double factnprec = (*sign2) * (*strain_rate) * (prectime - starttime);		
	if(curtime < starttime){
	  F = Id33;
	  Fprec = Id33;	  	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	} else {
	  F = Id33 + m_x_m * factm + n_x_n * factn;
	  Fprec = Id33 + m_x_m * factmprec + n_x_n * factnprec;	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	}
      } else if(mode == "triaxial"){
	Vec3d m = *direction1;
	Vec3d n = *direction2;
	Vec3d o = *direction3;		
	Mat3d m_x_m = tensor( m, m );
	Mat3d n_x_n = tensor( n, n );
	Mat3d o_x_o = tensor( o, o );		
	double factm = (*sign1) * (*strain_rate) * (curtime - starttime);
	double factn = (*sign2) * (*strain_rate) * (curtime - starttime);	
	double facto = (*sign3) * (*strain_rate) * (curtime - starttime);
	double factmprec = (*sign1) * (*strain_rate) * (prectime - starttime);
	double factnprec = (*sign2) * (*strain_rate) * (prectime - starttime);	
	double factoprec = (*sign3) * (*strain_rate) * (prectime - starttime);			
	if(curtime < starttime){
	  F = Id33;
	  Fprec = Id33;	  	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	} else {
	  F = Id33 + m_x_m * factm + n_x_n * factn + o_x_o * facto;
	  Fprec = Id33 + m_x_m * factmprec + n_x_n * factnprec + o_x_o * factoprec;	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	}
      } else if(mode == "shear"   ){
	Vec3d m = *direction1;
	Vec3d n = *direction2;	
	Mat3d m_x_n = tensor( m, n );
	double fact = (*strain_rate) * (curtime - starttime);
	double factprec = (*strain_rate) * (prectime - starttime);	
	if(curtime < starttime){
	  F = Id33;
	  Fprec = Id33;	  	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	} else {
	  F = Id33 + m_x_n * fact;
	  Fprec = Id33 + m_x_n * factprec;	  
	  xformlocal = F * inverse(Fprec) * predef;	  	  
	}
      } else{
	xformlocal = make_identity_matrix();
      }      

      macroscopic_strain = F;

      domain->set_xform( xformlocal * domain->xform());

      string interpolated_xform = onika::format_string("\t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n", (xform).m11, (xform).m12, (xform).m13, (xform).m21, (xform).m22, (xform).m23, (xform).m31, (xform).m32, (xform).m33);
      ldbg << "\n\tConstant strain-rate xform at time t = " << curtime << " s"<< std::endl;
      ldbg << interpolated_xform << std::endl;
    }

  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(constant_strain_rate_xform)
  {
   OperatorNodeFactory::instance()->register_factory(
    "xform_constant_strain_rate",
    make_compatible_operator< ConstantStrainRateXFormNode >
    );
  }

}


