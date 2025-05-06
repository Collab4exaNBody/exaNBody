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

#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <exanb/mesh/edge.h>
#include <onika/math/basic_types_operators.h>
#include <exanb/core/geometry.h>

namespace exanb
{

  struct EdgePointProjection
  {
      Vec3d m_proj;     // projected point
      double m_dist;    // point's signed ditance to edge's line
      double m_u;       // parametric coord (projection lies inside edge if u>=0 and u<=1)
      Vec3d m_normal;   // edge's unit normal vector pointing toward tested point
      bool m_inside;
  };

  ONIKA_HOST_DEVICE_FUNC
  inline EdgePointProjection project_edge_point( const Edge & e , const Vec3d & p )
  {
    using onika::math::norm;
    using onika::math::cross;
    using onika::math::dot;
    
    const auto dir = ( e[1] - e[0] );
    const auto edge_middle = ( e[0] + e[1] ) * 0.5;
    const auto middle_to_p = p - edge_middle;
    const auto edge_ortho = cross( middle_to_p , dir );
    auto plane_normal = cross( dir , edge_ortho );
    plane_normal = plane_normal / norm( plane_normal );
    const auto plane_d = - dot( plane_normal , edge_middle );
    const auto dist = dot( plane_normal , p ) + plane_d;
    
    const auto proj = p - ( plane_normal * dist );
    const auto u = dot( dir , proj - e[0] ) / norm( dir );

    return { proj , dist , u , plane_normal , u>=0 && u<=1 };
  }

}



/*****************
 *** unit test ***
 *****************/ 
#include <random>
#include <onika/test/unit_test.h>

ONIKA_UNIT_TEST(edge_math)
{
  using namespace onika;
  using namespace onika::math;
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -10.0 , 10.0 );

  for(int i=0;i<1000000;i++)
  {
    const Edge e = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const Vec3d p = {ud(gen),ud(gen),ud(gen)};
    const auto proj = project_edge_point( e, p );
    // std::cout << "edge=("<< e[0]<<","<<e[1]<<") , p="<<p<<" , proj="<<proj.m_proj<<" , dist="<<proj.m_dist<<std::endl;
    ONIKA_TEST_ASSERT( norm( proj.m_proj + proj.m_dist * proj.m_normal - p ) < 1e-12 );
    ONIKA_TEST_ASSERT( std::abs( norm( p - proj.m_proj ) - proj.m_dist ) < 1e-12 );
    ONIKA_TEST_ASSERT( std::abs( dot( p - proj.m_proj , e[1] - e[0] ) ) < 1e-12 );
  }

}
