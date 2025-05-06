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
#include <exanb/mesh/triangle.h>
#include <exanb/mesh/edge.h>
#include <onika/math/basic_types_operators.h>
#include <exanb/core/geometry.h>

namespace exanb
{

  ONIKA_HOST_DEVICE_FUNC
  inline double area( const Triangle & t )
  {
    using onika::math::norm;
    using onika::math::cross;
    const auto AB = t[1] - t[0];
    const auto AC = t[2] - t[0];
    return norm( cross(AB,AC) ) * 0.5;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline onika::math::Vec3d normal( const Triangle & t )
  {
    using onika::math::norm;
    using onika::math::cross;
    const auto AB = t[1] - t[0];
    const auto AC = t[2] - t[0];
    const auto AB_AC = cross(AB,AC);
    auto L = norm(AB_AC);
    if( L > 0 ) L = 1 / L;
    return AB_AC * L;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline onika::math::AABB bounds( const Triangle & t )
  {
    using onika::cuda::min;
    using onika::cuda::max;
    return { { min(min(t[0].x,t[1].x),t[2].x) , min(min(t[0].y,t[1].y),t[2].y) , min(min(t[0].z,t[1].z),t[2].z) } , { max(max(t[0].x,t[1].x),t[2].x) , max(max(t[0].y,t[1].y),t[2].y) , max(max(t[0].z,t[1].z),t[2].z) } };
  }

  struct TrianglePointProjection
  {
      Vec3d m_proj;     // projected point
      double m_dist;         // point's signed ditance to triangle's plane
      double m_u, m_v, m_w;            // coordonnées barycentriques
      Vec3d m_normal;          // triangle's unit normal vector
      bool m_behind;         // is point behind triangle ?
      bool m_inside;         // is projected point contained inside triangle ?
  };

  ONIKA_HOST_DEVICE_FUNC
  inline onika::math::Vec3d triangle_point_barycentric_coords(const Triangle& t, const Vec3d& p)
  {
    using onika::math::dot;

    const auto& a = t[0];
    const auto& b = t[1];
    const auto& c = t[2];
    
    const auto v0 = b - a;
    const auto v1 = c - a;
    const auto v2 = p - a;
    
    const auto d00 = dot(v0, v0);
    const auto d01 = dot(v0, v1);
    const auto d11 = dot(v1, v1);
    const auto d20 = dot(v2, v0);
    const auto d21 = dot(v2, v1);
    const auto denom = d00 * d11 - d01 * d01;
    const auto v = (d11 * d20 - d01 * d21) / denom;
    const auto w = (d00 * d21 - d01 * d20) / denom;
    const auto u = 1 - v - w;
    return { u , v , w };
  }

  ONIKA_HOST_DEVICE_FUNC
  inline TrianglePointProjection project_triangle_point( const Triangle & t , const Vec3d & p )
  {
    using onika::math::norm;
    using onika::math::cross;
    using onika::math::dot;
    
    // triangle unit normal vector
    const auto N = normal( t );
    
    // prejected point onto triangle's plane
    const auto rel_p = p - t[0];
    const auto d = dot( rel_p , N );
    const auto proj_p = p - ( d * N );
    
    const auto [ u , v , w ] = triangle_point_barycentric_coords( t , proj_p );
    
    return { proj_p, d, u, v, w, N , d < 0 , ( u>=0 && u<=1 && v>=0 && v<=1 && w>=0 && w<=1 ) };
  }

  ONIKA_HOST_DEVICE_FUNC
  inline bool intersect(const onika::math::AABB & box, const Triangle& tri)
  {
    using onika::cuda::min;
    using onika::cuda::max;
    using onika::math::Vec3d;
    
	  const auto tri_bounds = bounds( tri );

	  if( is_empty( intersection(box,tri_bounds) ) ) return false;

	  // triangle-normal and edges
	  const auto n = normal( tri );
	  const auto v0 = tri[0];
	  const auto v1 = tri[1];
	  const auto v2 = tri[2];
	  const auto edge0 = v1 - v0;
	  const auto edge1 = v2 - v1;
	  const auto edge2 = v0 - v2;

	  // p & delta-p
	  const auto p  = tri_bounds.bmin;
	  const auto dp = tri_bounds.bmax - p;

	  // test for triangle-plane/box overlap
	  Vec3d c = { n.x > 0. ? dp.x : 0., n.y > 0. ? dp.y : 0., n.z > 0. ? dp.z : 0. };

	  const auto d1 = dot(n, c - v0);
	  const auto d2 = dot(n, dp - c - v0);

	  if( (dot(n, p) + d1) * (dot(n, p) + d2) > 0. ) return false;

	  // xy-plane projection-overlap
	  const auto xym = (n.z < 0. ? -1. : 1.);
	  const auto ne0xy = Vec3d{-edge0.y, edge0.x, 0.} * xym;
	  const auto ne1xy = Vec3d{-edge1.y, edge1.x, 0.} * xym;
	  const auto ne2xy = Vec3d{-edge2.y, edge2.x, 0.} * xym;

	  const auto v0xy = Vec3d{v0.x, v0.y, 0.};
	  const auto v1xy = Vec3d{v1.x, v1.y, 0.};
	  const auto v2xy = Vec3d{v2.x, v2.y, 0.};

	  const auto de0xy = -dot(ne0xy, v0xy) + max(0., dp.x * ne0xy.x) + max(0., dp.y * ne0xy.y);
	  const auto de1xy = -dot(ne1xy, v1xy) + max(0., dp.x * ne1xy.x) + max(0., dp.y * ne1xy.y);
	  const auto de2xy = -dot(ne2xy, v2xy) + max(0., dp.x * ne2xy.x) + max(0., dp.y * ne2xy.y);

	  const Vec3d pxy = { p.x, p.y, 0. };

	  if( (dot(ne0xy, pxy) + de0xy) < 0. || (dot(ne1xy, pxy) + de1xy) < 0. || (dot(ne2xy, pxy) + de2xy) < 0. ) return false;

	  // yz-plane projection overlap
	  const auto yzm = (n.x < 0. ? -1. : 1.);
	  const auto ne0yz = Vec3d{-edge0.z, edge0.y, 0. } * yzm;
	  const auto ne1yz = Vec3d{-edge1.z, edge1.y, 0. } * yzm;
	  const auto ne2yz = Vec3d{-edge2.z, edge2.y, 0. } * yzm;

	  const Vec3d v0yz = { v0.y, v0.z, 0. };
	  const Vec3d v1yz = { v1.y, v1.z, 0. };
	  const Vec3d v2yz = { v2.y, v2.z, 0. };

	  const auto de0yz = -dot(ne0yz, v0yz) + max(0., dp.y * ne0yz.x) + max(0., dp.z * ne0yz.y);
	  const auto de1yz = -dot(ne1yz, v1yz) + max(0., dp.y * ne1yz.x) + max(0., dp.z * ne1yz.y);
	  const auto de2yz = -dot(ne2yz, v2yz) + max(0., dp.y * ne2yz.x) + max(0., dp.z * ne2yz.y);

	  const Vec3d pyz = { p.y, p.z, 0. };

	  if( (dot(ne0yz, pyz) + de0yz) < 0. || (dot(ne1yz, pyz) + de1yz) < 0. || (dot(ne2yz, pyz) + de2yz) < 0. ) return false;

	  // zx-plane projection overlap
	  const auto zxm = (n.y < 0. ? -1. : 1.);
	  const auto ne0zx = Vec3d{-edge0.x, edge0.z, 0.} * zxm;
	  const auto ne1zx = Vec3d{-edge1.x, edge1.z, 0.} * zxm;
	  const auto ne2zx = Vec3d{-edge2.x, edge2.z, 0.} * zxm;

	  const Vec3d v0zx = {v0.z, v0.x, 0.};
	  const Vec3d v1zx = {v1.z, v1.x, 0.};
	  const Vec3d v2zx = {v2.z, v2.x, 0.};

	  const auto de0zx = -dot(ne0zx, v0zx) + max(0., dp.y * ne0zx.x) + max(0., dp.z * ne0zx.y);
	  const auto de1zx = -dot(ne1zx, v1zx) + max(0., dp.y * ne1zx.x) + max(0., dp.z * ne1zx.y);
	  const auto de2zx = -dot(ne2zx, v2zx) + max(0., dp.y * ne2zx.x) + max(0., dp.z * ne2zx.y);

	  const Vec3d pzx = { p.z, p.x, 0. };

	  if( (dot(ne0zx, pzx) + de0zx) < 0. || (dot(ne1zx, pzx) + de1zx) < 0. || (dot(ne2zx, pzx) + de2zx) < 0. ) return false;

	  return true;
  }

  struct TriangleEdgeIntersection
  {
    Vec3d m_intersection;
    double m_tri_u, m_tri_v, m_tri_w;
    double m_edge_u;
    bool m_intersect;
  };

  ONIKA_HOST_DEVICE_FUNC
  inline TriangleEdgeIntersection triangle_edge_intersection( const Triangle & t , const Edge & e )
  {
    using onika::math::norm;
    using onika::math::cross;
    using onika::math::dot;
    
    // triangle unit normal vector
    const auto plane_n = normal( t );
    const auto plane_d = - dot( plane_n , t[0] );
    
    // compute triangle's plane / edge intersection
    const auto d0 = dot( plane_n , e[0] ) + plane_d;
    const auto d1 = dot( plane_n , e[1] ) + plane_d;
    const auto delta_d = d1 - d0;
    const auto edge_u = std::abs(delta_d)>1e-12 ? -d0 / delta_d : 0;
    
    const auto plane_intersection = e[0] + edge_u * ( e[1] - e[0] ); 
    const auto plane_intersection_dist = dot( plane_n , plane_intersection ) + plane_d;        
    const auto [ u , v , w ] = triangle_point_barycentric_coords( t , plane_intersection );
    return { plane_intersection , u,v,w, edge_u , std::abs(plane_intersection_dist)<1e-9 && edge_u>=0 && edge_u<=1 && u>=0 && u<=1 && v>=0 && v<=1 && w>=0 && w<=1 };
  }

}


/*****************
 *** unit test ***
 *****************/ 
#include <random>
#include <onika/test/unit_test.h>

ONIKA_UNIT_TEST(triangle_math)
{
  using namespace onika;
  using namespace onika::math;
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -10.0 , 10.0 );

  for(int i=0;i<1000000;i++)
  {
    const Triangle t = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)}  , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const Vec3d p = {ud(gen),ud(gen),ud(gen)};

    const auto proj = project_triangle_point( t, p );
    const auto plane_d = - dot( proj.m_normal , t[0] );
    const auto plane_dist = dot( proj.m_normal , p ) + plane_d;
    const auto plane_proj_dist = dot( proj.m_normal , proj.m_proj ) + plane_d;

    ONIKA_TEST_ASSERT( std::abs( proj.m_u + proj.m_v + proj.m_w - 1 ) < 1e-12 );
    ONIKA_TEST_ASSERT( std::abs( plane_dist - proj.m_dist )  < 1.e-12 );
    ONIKA_TEST_ASSERT( std::abs( plane_proj_dist )  < 1.e-12 );
    ONIKA_TEST_ASSERT( proj.m_inside == ( proj.m_u>=0 && proj.m_u<=1 && proj.m_v>=0 && proj.m_v<=1 && proj.m_w>=0 && proj.m_w<=1 ) );

    if( proj.m_inside )
    {
      const auto bary = proj.m_u * t[0] + proj.m_v * t[1] + proj.m_w * t[2];
      ONIKA_TEST_ASSERT( norm(bary - proj.m_proj) < 1.e-9 );
      ONIKA_TEST_ASSERT( std::abs( norm( p - bary ) - std::abs(proj.m_dist) ) < 1.e-9 );
      ONIKA_TEST_ASSERT( std::abs( dot( p - bary , proj.m_normal ) - proj.m_dist ) < 1.e-9 );
    }
  }

  for(int i=0;i<1000000;i++)
  {
    const Triangle t = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const Edge e = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const auto inter = triangle_edge_intersection( t, e );
    const auto edge_bary = e[0] + ( inter.m_edge_u * (e[1] - e[0]) );
    const auto tri_bary = inter.m_tri_u * t[0] + inter.m_tri_v * t[1] + inter.m_tri_w * t[2];
    const auto plane_n = normal( t );
    const auto plane_d = - dot( t[0] , plane_n );

    if( inter.m_intersect )
    {
      ONIKA_TEST_ASSERT( (inter.m_tri_u+inter.m_tri_v+inter.m_tri_w-1) < 1e-12 );
      ONIKA_TEST_ASSERT( norm( inter.m_intersection - edge_bary ) < 1e-9 );
      ONIKA_TEST_ASSERT( norm( inter.m_intersection - tri_bary ) < 1e-9 );
      ONIKA_TEST_ASSERT( dot( plane_n , inter.m_intersection ) + plane_d < 1e-9 );
    }

  }

}
