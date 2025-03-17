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
#include <exanb/core/domain.h>

#include <string>
#include <onika/string_utils.h>

using namespace std;

namespace exanb
{

  struct InterpolatedByPartsXFormNode : public OperatorNode
  {
    using XFormVec = std::vector<Mat3d>;
    using TimeVec = std::vector<double>;

    ADD_SLOT( XFormVec , xform_serie , INPUT , REQUIRED );
    ADD_SLOT( TimeVec  , time_serie  , INPUT , REQUIRED );
    ADD_SLOT( long     , timestep    , INPUT , REQUIRED );
    ADD_SLOT( double   , dt          , INPUT , REQUIRED );
    ADD_SLOT( Domain   , domain      , INPUT_OUTPUT);

    template<typename YFunc>
    static inline double interpolate( const std::vector<double>& X , double ix, YFunc yfunc )
    {
      assert( std::is_sorted( X.begin() , X.end() ) );
      size_t N = X.size();
      
      if( N == 0 )
      {
        return 0.0;
      }
      if( N == 1 )
      {
        return yfunc(0);
      }
      if( ix < X[0] )
      {
        return yfunc(0);
      }
      
      size_t k = 0;
      while( ix > X[k+1] && k<(N-2) ) { ++k; }
      
      if( ix > X[k+1] )
      {
        return yfunc(k+1);
      }

      double t = (ix - X[k]) / ( X[k+1] - X[k] );
      assert( t>=0 && t<=1.0 );

      return yfunc(k)*(1.-t) + yfunc(k+1)*t;
    }

    inline void execute () override final
    {
      if( xform_serie->size() != time_serie->size() )
      {
        lerr << "number of time values does not match number of transforms." << std::endl;
        std::abort();
      }

      double prectime = (*dt) * (*timestep - 1);
      double curtime = (*dt) * (*timestep);
      const XFormVec& xfvec = *xform_serie;

      Mat3d Fprec;
      Fprec.m11 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m11; } );
      Fprec.m12 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m12; } );
      Fprec.m13 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m13; } );

      Fprec.m21 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m21; } );
      Fprec.m22 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m22; } );
      Fprec.m23 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m23; } );

      Fprec.m31 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m31; } );
      Fprec.m32 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m32; } );
      Fprec.m33 = interpolate( *time_serie , prectime , [&xfvec](size_t i)->double { return xfvec[i].m33; } );

      Mat3d F;
      F.m11 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m11; } );
      F.m12 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m12; } );
      F.m13 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m13; } );

      F.m21 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m21; } );
      F.m22 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m22; } );
      F.m23 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m23; } );

      F.m31 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m31; } );
      F.m32 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m32; } );
      F.m33 = interpolate( *time_serie , curtime , [&xfvec](size_t i)->double { return xfvec[i].m33; } );      

      Mat3d xformlocal;
      xformlocal = F * inverse(Fprec);
      domain->set_xform( xformlocal * domain->xform() );
  
      string interpolated_xform = onika::format_string("\t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n", F.m11, F.m12, F.m13, F.m21, F.m22, F.m23, F.m31, F.m32, F.m33);
      ldbg << "\n\tInterpolated XForm at time T = " << curtime << std::endl;
      ldbg << interpolated_xform << std::endl;
    }

  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(interpolated_byparts_xform)
  {
   OperatorNodeFactory::instance()->register_factory(
    "xform_time_interpolate_byparts",
    make_compatible_operator< InterpolatedByPartsXFormNode >
    );
  }

}


