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
      double u_bary, v_bary, w_bary;            // coordonnées barycentriques
      Vec3d normal;          // triangle's unit normal vector
      bool m_behind;         // is point behind triangle ?
      bool m_inside;         // is projected point contained inside triangle ?
  };

  ONIKA_HOST_DEVICE_FUNC
  inline TrianglePointProjection project_triangle_point( const Triangle & t , const Vec3d & p )
  {
    using onika::math::norm;
    using onika::math::cross;
    using onika::math::dot;
    
    // triangle unit normal vector
    const auto AB = t[1] - t[0];
    const auto AC = t[2] - t[0];
    const auto N = normal( t );
    
    // prejected point onto triangle's plane
    const auto v = p - t[0];
    const auto d = dot(v,N);
    const auto pv = v - ( d * N );

    // compute barycentric coordinates
    const auto d00 = dot(AB,AB);
    const auto d01 = dot(AB,AC);
    const auto d11 = dot(AC,AC);
    const auto d20 = dot(pv,AB);
    const auto d21 = dot(pv,AC);

    const auto denom = d00 * d11 - d01 * d01;
    const auto v_bary = (d11 * d20 - d01 * d21) / denom;
    const auto w_bary = (d00 * d21 - d01 * d20) / denom;
    const auto u_bary = 1.0 - v_bary - w_bary;

    return { pv, d, u_bary, v_bary, w_bary, N , d < 0 , ( u_bary >= 0.0 && v_bary >= 0.0 && w_bary >= 0.0 && u_bary <= 1.0 && v_bary <= 1.0 && w_bary <= 1.0 ) };
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

}


