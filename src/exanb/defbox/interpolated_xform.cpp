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
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>

#include <exanb/core/spline.h>

namespace exanb
{

  struct InterpolatedXFormNode : public OperatorNode
  {
    using XFormVec = std::vector<Mat3d>;
    using TimeVec = std::vector<double>;

    ADD_SLOT( XFormVec , xform_serie  , INPUT);
    ADD_SLOT( TimeVec  , time_serie   , INPUT);
    ADD_SLOT( double   , physical_time, INPUT);
    ADD_SLOT( Domain   , domain       , INPUT_OUTPUT);

    template<typename YFunc>
    static inline double interpolate( const std::vector<double>& X , double ix, YFunc yfunc )
    {
      assert( std::is_sorted( X.begin() , X.end() ) );
      size_t N = X.size();
      std::vector<double> Y(N,0.0);
      for(size_t i=0;i<N;i++) { Y[i] = yfunc(i); }
      exanb::Spline s;
      s.set_points(X,Y);    // X needs to be sorted, strictly increasing
      return s.eval(ix);
    }

    inline void execute ()  override final
    {
      if( xform_serie->size() != time_serie->size() )
      {
        lerr << "number of time values does not match number of transforms." << std::endl;
        std::abort();
      }

      const XFormVec& xfvec = *xform_serie;

      Mat3d mat;
      mat.m11 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m11; } );
      mat.m12 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m12; } );
      mat.m13 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m13; } );

      mat.m21 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m21; } );
      mat.m22 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m22; } );
      mat.m23 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m23; } );

      mat.m31 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m31; } );
      mat.m32 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m32; } );
      mat.m33 = interpolate( *time_serie , *physical_time , [&xfvec](size_t i)->double { return xfvec[i].m33; } );
      
      domain->set_xform( mat );

      std::string interpolated_xform = onika::format_string("\t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n \t | %-5.4e \t %-5.4e \t %-5.4e | \n", 
                                                mat.m11, mat.m12, mat.m13, mat.m21, mat.m22, mat.m23, mat.m31, mat.m32, mat.m33);
                                                
      ldbg << "\n\tInterpolated XForm at time T = " << *physical_time << std::endl;
      ldbg << interpolated_xform << std::endl;      
    }

  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory(
    "xform_time_interpolate",
    make_compatible_operator< InterpolatedXFormNode >
    );
  }

}


