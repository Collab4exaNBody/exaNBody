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
  inline int plane_polygon_clip(const Vec3d& plane_n, const double plane_d, const onika::math::Vec3d * in_verts , int n , onika::math::Vec3d * out_verts , int max_n)
  {
    using onika::math::dot;
    int n_out_verts = 0;

    if( n <= 0 ) return n;
    for(int i=0;i<n;i++)
    {
      const auto e0 = in_verts[i];
      const auto e1 = in_verts[(i+1)%n];
      const auto d0 = dot( plane_n , e0 ) + plane_d;
      const auto d1 = dot( plane_n , e1 ) + plane_d;
      if( d0 >= 0 )
      {
        assert( n_out_verts < max_n );
        out_verts[n_out_verts++] = e0;
      }
      if( d0*d1 < 0 )
      {
        assert( n_out_verts < max_n );
        const auto delta_d = d1 - d0;
        const auto d01_norm_factor = ( d0 != 0 || d1 != 0 ) ? ( abs(d0) + abs(d1) ) : 1.0 ;
        const auto edge_u = abs(delta_d/d01_norm_factor)>1e-12 ? ( -d0 / delta_d ) : 0.5;
        out_verts[n_out_verts++] = e0 + edge_u * ( e1 - e0 );
      }
    }
    return n_out_verts;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline bool intersect(const onika::math::AABB & box, const Triangle& tri)
  {
    using onika::cuda::min;
    using onika::cuda::max;
    using onika::math::Vec3d;
    static constexpr size_t MAX_POLY_VERTS = 12;
    
    Vec3d poly_buf_a[MAX_POLY_VERTS] = { tri[0] , tri[1] , tri[2] , };
    auto * poly_a = poly_buf_a;
    int n_verts = 3;
    Vec3d poly_buf_b[MAX_POLY_VERTS];
    auto * poly_b = poly_buf_b;
    
    const double box_min[3] = { box.bmin.x , box.bmin.y , box.bmin.z };
    const double box_max[3] = { box.bmax.x , box.bmax.y , box.bmax.z };

    for(int i=0;i<6;i++)
    {
      const int axis = i/2;
      const bool lower_side = ( (i%2) == 0 );
      double plane_vec[3] = { 0 , 0 , 0 };
      plane_vec[ axis ] = lower_side ? 1.0 : -1.0;
      const Vec3d plane_n = { plane_vec[0] , plane_vec[1] , plane_vec[2] };
      const double plane_d = lower_side ? - box_min[axis] : box_max[axis];
      n_verts = plane_polygon_clip( plane_n, plane_d, poly_a, n_verts, poly_b, MAX_POLY_VERTS );
      if( n_verts == 0 ) return false;
      auto * tmp = poly_b; poly_b = poly_a; poly_a = tmp;
    }

    //printf("clipped polygon has %d vertices\n",n_verts);
	  return true;
  }

  struct TriangleEdgeIntersection
  {
    Vec3d m_intersection;
    double m_typical_length;
    double m_tri_u, m_tri_v, m_tri_w;
    double m_edge_u;
    bool m_plane_edge_intersect;
    bool m_inside_trianle;
    bool m_intersect;
  };

  ONIKA_HOST_DEVICE_FUNC
  inline TriangleEdgeIntersection triangle_edge_intersection( const Triangle & t , const Edge & e )
  {
    using onika::math::norm;
    using onika::math::cross;
    using onika::math::dot;
    using onika::cuda::max;
    using onika::cuda::min;

    // triangle unit normal vector
    const auto plane_n = normal( t );

    const Vec3d box_min = { min( min(min(t[0].x,t[1].x),t[2].x) , min(e[0].x,e[1].x) )
                          , min( min(min(t[0].y,t[1].y),t[2].y) , min(e[0].y,e[1].y) )
                          , min( min(min(t[0].z,t[1].z),t[2].z) , min(e[0].z,e[1].z) ) };
    const Vec3d box_max = { max( max(max(t[0].x,t[1].x),t[2].x) , max(e[0].x,e[1].x) )
                          , max( max(max(t[0].y,t[1].y),t[2].y) , max(e[0].y,e[1].y) )
                          , max( max(max(t[0].z,t[1].z),t[2].z) , max(e[0].z,e[1].z) ) };
    const auto typical_length = norm( box_max - box_min );
    if( typical_length == 0.0 ) return { t[0]+plane_n, 1.0, -1,-1,-1, -1, false, false, false };

    const auto plane_d = - dot( plane_n , t[0] );
    
    // compute triangle's plane / edge intersection
    const auto d0 = dot( plane_n , e[0] ) + plane_d;
    const auto d1 = dot( plane_n , e[1] ) + plane_d;
    const auto delta_d = d1 - d0;
    const auto d01_norm_factor = ( d0 != 0 || d1 != 0 ) ? ( abs(d0) + abs(d1) ) : 1.0 ;
    const auto edge_u = std::abs(delta_d/d01_norm_factor)>1e-12 ? ( -d0 / delta_d ) : 0.5;
    
    const auto plane_intersection = e[0] + edge_u * ( e[1] - e[0] ); 
    const auto plane_intersection_dist = dot( plane_n , plane_intersection ) + plane_d;
    const auto [ u , v , w ] = triangle_point_barycentric_coords( t , plane_intersection );
    
    const bool plane_edge_intersect = ( edge_u>=0 ) && ( edge_u<=1 ) && ( std::abs(plane_intersection_dist/typical_length) < 1e-9 );
    const bool inside_trianle = ( u>=0 ) && ( u<=1 ) && ( v>=0 ) && ( v<=1 ) && ( w>=0 ) && ( w<=1 );
    return { plane_intersection, typical_length, u,v,w, edge_u, plane_edge_intersect, inside_trianle, plane_edge_intersect && inside_trianle };
  }

}


/*****************
 *** unit test ***
 *****************/ 
#include <random>
#include <onika/test/unit_test.h>

ONIKA_UNIT_TEST(project_triangle_point)
{
  using namespace onika;
  using namespace onika::math;
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -10.0 , 10.0 );

# pragma omp parallel for schedule(static)
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
}

ONIKA_UNIT_TEST(triangle_edge_intersection)
{
  using namespace onika;
  using namespace onika::math;
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -10.0 , 10.0 );

# pragma omp parallel for schedule(static)
  for(int i=0;i<1000000;i++)
  {
    const Triangle t = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const Edge e = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const auto inter = triangle_edge_intersection( t, e );    
    const auto plane_n = normal( t );
    const auto plane_d = - dot( t[0] , plane_n );
    if( inter.m_intersect )
    {
      const auto edge_bary = e[0] + ( inter.m_edge_u * (e[1] - e[0]) );
      const auto tri_bary = inter.m_tri_u * t[0] + inter.m_tri_v * t[1] + inter.m_tri_w * t[2];
      ONIKA_TEST_ASSERT( inter.m_plane_edge_intersect && inter.m_inside_trianle );
      ONIKA_TEST_ASSERT( (inter.m_tri_u+inter.m_tri_v+inter.m_tri_w-1) < 1e-12 );
      ONIKA_TEST_ASSERT( norm( inter.m_intersection - edge_bary ) < 1e-9 );
      ONIKA_TEST_ASSERT( norm( inter.m_intersection - tri_bary ) < 1e-9 );
      ONIKA_TEST_ASSERT( dot( plane_n , inter.m_intersection ) + plane_d < 1e-9 );
    }
    else
    {
      ONIKA_TEST_ASSERT( ! inter.m_plane_edge_intersect || ! inter.m_inside_trianle );      
      if( inter.m_plane_edge_intersect )
      {
        ONIKA_TEST_ASSERT( inter.m_tri_u<0 || inter.m_tri_u>1 || inter.m_tri_v<0 || inter.m_tri_v>1 || inter.m_tri_w<0 || inter.m_tri_w>1 );
      }
      if( inter.m_inside_trianle )
      {
        ONIKA_TEST_ASSERT( inter.m_edge_u<0 || inter.m_edge_u>1 || ( ( dot( plane_n , inter.m_intersection ) + plane_d ) / inter.m_typical_length ) > 1e-9 );
      }
    }
  }
}

ONIKA_UNIT_TEST(triangle_box_intersection)
{
  using namespace onika;
  using namespace onika::math;
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -10.0 , 10.0 );

# pragma omp parallel for schedule(static)
  for(int i=0;i<100000;i++)
  {
    const Triangle t = { Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} , Vec3d{ud(gen),ud(gen),ud(gen)} };
    const Vec3d p1 = {ud(gen),ud(gen),ud(gen)};
    const Vec3d p2 = {ud(gen),ud(gen),ud(gen)};
    const AABB box = { Vec3d{ std::min(p1.x,p2.x) , std::min(p1.y,p2.y) , std::min(p1.z,p2.z) } , Vec3d{ std::max(p1.x,p2.x) , std::max(p1.y,p2.y) , std::max(p1.z,p2.z) } };
    
    const AABB all_bounds = extend(box,bounds(t));
    const auto bounds_length = norm( all_bounds.bmax - all_bounds.bmin );
    ONIKA_TEST_ASSERT( bounds_length > 0 );
        
    double min_dist = 999999.0;
    for(double u = 0.0 ;  u   <= 1.0 ; u += 0.001 )
    for(double v = 0.0 ; (u+v)<= 1.0 ; v += 0.001 )
    {
      const double w = 1.0 - u - v ;
      //ONIKA_TEST_ASSERT( std::abs( u+v+w - 1 ) < 1e-12 );
      const auto p = t[0]*u + t[1]*v + t[2]*w;
      const auto dist = min_distance_between(p,box);
      if( dist < min_dist )
      {
        min_dist = dist;
      }
    }
    
    const bool inter = intersect(box,t);
    ONIKA_TEST_ASSERT( ( inter && (min_dist/bounds_length<1e-3) ) || ( !inter && (min_dist>0) ) );
  }
}
