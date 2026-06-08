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

#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <cmath>
#include <algorithm>
#include <cassert>

#include <onika/cuda/cuda_math.h>

// TODO: use integer arithmetic when possible

namespace exanb
{
  ONIKA_HOST_DEVICE_FUNC inline double distance2(Vec3d a, Vec3d b)
  {
    b.x -= a.x;
    b.y -= a.y;
    b.z -= a.z;
    return b.x*b.x + b.y*b.y + b.z*b.z;
  }

  inline double distance(Vec3d a, Vec3d b)
  {
    return std::sqrt( distance2(a,b) );
  }

  inline Vec3d grid_cell_position(IJK p)
  {
    return Vec3d{ static_cast<double>(p.i), static_cast<double>(p.j), static_cast<double>(p.k) };
  }

  inline AABB grid_cell_bounds(IJK p)
  {
    return AABB{ grid_cell_position(p) , grid_cell_position(p+1) };
  }

  ONIKA_HOST_DEVICE_FUNC inline AABB transform(AABB b, double s, Vec3d o)
  {
    return AABB{ {b.bmin.x*s+o.x, b.bmin.y*s+o.y, b.bmin.z*s+o.z} , {b.bmax.x*s+o.x, b.bmax.y*s+o.y, b.bmax.z*s+o.z} };
  }

  inline AABB intersection(const AABB& b1, const AABB& b2)
  {
    // first compute intersection of b1 and b2;
    double xmin = std::max( b1.bmin.x, b2.bmin.x );
    double ymin = std::max( b1.bmin.y, b2.bmin.y );
    double zmin = std::max( b1.bmin.z, b2.bmin.z );

    double xmax = std::min( b1.bmax.x, b2.bmax.x );
    double ymax = std::min( b1.bmax.y, b2.bmax.y );
    double zmax = std::min( b1.bmax.z, b2.bmax.z );

    return AABB { {xmin,ymin,zmin} , {xmax,ymax,zmax} };
  }


  // return true if b is the intersection of 2 disjoints areas, e.g. min > max for one of the components
  ONIKA_HOST_DEVICE_FUNC inline bool is_empty(const AABB& b)
  {    
    return b.bmin.x>=b.bmax.x || b.bmin.y>=b.bmax.y || b.bmin.z>=b.bmax.z;
  }

  // for backward compatibility
  [[deprecated]] ONIKA_HOST_DEVICE_FUNC inline bool is_nil(const AABB& b) { return is_empty(b); }

  inline double max_distance_inside(const AABB& b)
  {
    double x = b.bmax.x - b.bmin.x;
    double y = b.bmax.y - b.bmin.y;
    double z = b.bmax.z - b.bmin.z;
    return std::sqrt( x*x + y*y + z*z );
  }
  
  inline void range_reorder_min_max(double& amin, double& amax)
  {
    if( amin > amax ) { std::swap( amin, amax ); }
  }

  inline void reorder_min_max(AABB& b)
  {
    range_reorder_min_max( b.bmin.x , b.bmax.x );
    range_reorder_min_max( b.bmin.y , b.bmax.y );
    range_reorder_min_max( b.bmin.z , b.bmax.z );
  }

  ONIKA_HOST_DEVICE_FUNC inline bool range_intersect(double amin, double amax, double bmin, double bmax)
  {
    return amax >= bmin && amin <= bmax;
  }

  ONIKA_HOST_DEVICE_FUNC inline double range_min_dist(double amin, double amax, double bmin, double bmax)
  {
    assert(amin<=amax && bmin<=bmax);
    if( ! range_intersect( amin, amax, bmin, bmax ) )
    {
      if( amin < bmin ) { return bmin - amax; }
      else { return amin - bmax; } 
    }
    else
    {
      return 0.0;
    }
  }

  // return minimal distance of 2 points p1 and p2, given that p1 is inside b1 and p2 is inside b2
  ONIKA_HOST_DEVICE_FUNC inline double min_distance2_between(const AABB& b1, const AABB& b2)
  {
    double x = range_min_dist( b1.bmin.x, b1.bmax.x, b2.bmin.x, b2.bmax.x );
    double y = range_min_dist( b1.bmin.y, b1.bmax.y, b2.bmin.y, b2.bmax.y );
    double z = range_min_dist( b1.bmin.z, b1.bmax.z, b2.bmin.z, b2.bmax.z );
    return x*x + y*y + z*z;
  }

  inline double min_distance_between(AABB b1, AABB b2)
  {
    return std::sqrt( min_distance2_between(b1,b2) );
  }

  /*
    return the square of the distance between point 'p' and the closest point in box 'b'
  */
  ONIKA_HOST_DEVICE_FUNC inline double min_distance2_between(const Vec3d& p, const AABB& b)
  {
    double dx = 0.;
    if( p.x < b.bmin.x) { dx = b.bmin.x - p.x; }
    else if( p.x > b.bmax.x ) { dx = p.x - b.bmax.x; }

    double dy = 0.;
    if( p.y < b.bmin.y) { dy = b.bmin.y - p.y; }
    else if( p.y > b.bmax.y ) { dy = p.y - b.bmax.y; }

    double dz = 0.;
    if( p.z < b.bmin.z) { dz = b.bmin.z - p.z; }
    else if( p.z > b.bmax.z ) { dz = p.z - b.bmax.z; }
    
    return dx*dx + dy*dy + dz*dz;
  }

  inline double min_distance_between(Vec3d p, AABB b)
  {
    return std::sqrt( min_distance2_between(p,b) );
  }

  ONIKA_HOST_DEVICE_FUNC inline double range_max_dist(double amin, double amax, double bmin, double bmax)
  {
    using onika::cuda::max;
    using onika::cuda::min;
    assert(amin<=amax && bmin<=bmax);
    return max( amax, bmax ) - min( amin, bmin );
  }

  ONIKA_HOST_DEVICE_FUNC inline double max_distance2_between(AABB b1, AABB b2)
  {
    double x = range_max_dist( b1.bmin.x, b1.bmax.x, b2.bmin.x, b2.bmax.x );
    double y = range_max_dist( b1.bmin.y, b1.bmax.y, b2.bmin.y, b2.bmax.y );
    double z = range_max_dist( b1.bmin.z, b1.bmax.z, b2.bmin.z, b2.bmax.z );
    return x*x + y*y + z*z;
  }

  inline double max_distance_between(AABB b1, AABB b2)
  {
    return std::sqrt( max_distance2_between(b1,b2) );
  }

  ONIKA_HOST_DEVICE_FUNC inline bool in_range(double rmin, double rmax, double x)
  {
    assert( rmin <= rmax );
    return x>=rmin && x<=rmax;
  }

  // to check if a particle is inside a cell, use this method with epsilon2 = 1.5e-30 * CellSize^2
  // E.g. call is_inside_threshold( grid.cell_bounds(ijk) , r , grid.epsilon_cell_size2() );
  ONIKA_HOST_DEVICE_FUNC inline bool is_inside_threshold(AABB b, Vec3d p, double epsilon2=1.5e-30)
  {
    double d2 = min_distance2_between(p,b);
    return d2 < epsilon2;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_inside(const AABB& r, const Vec3d& p)
  {
    return in_range(r.bmin.x,r.bmax.x,p.x)
        && in_range(r.bmin.y,r.bmax.y,p.y)
        && in_range(r.bmin.z,r.bmax.z,p.z);
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_inside_exclude_upper(AABB r, Vec3d p)
  {
    return ( p.x >= r.bmin.x && p.x < r.bmax.x )
        && ( p.y >= r.bmin.y && p.y < r.bmax.y )
        && ( p.z >= r.bmin.z && p.z < r.bmax.z );
  }

  // test if b is included in a (if a contains whole of b)
  ONIKA_HOST_DEVICE_FUNC inline bool is_inside(AABB a, AABB b)
  {
    return is_inside(a,b.bmin) && is_inside(a,b.bmax);
  }

  inline AABB extend(AABB r, Vec3d p)
  {
    return AABB{
      { std::min(r.bmin.x,p.x) , std::min(r.bmin.y,p.y) , std::min(r.bmin.z,p.z) } ,
      { std::max(r.bmax.x,p.x) , std::max(r.bmax.y,p.y) , std::max(r.bmax.z,p.z) } };
  }

  ONIKA_HOST_DEVICE_FUNC inline AABB enlarge(AABB r, double d)
  {
    return AABB{ r.bmin - d , r.bmax + d };
  }


  inline AABB extend(AABB b1, AABB b2)
  {
    AABB r = extend(b1,b2.bmin);
    return extend(r,b2.bmax);
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d bounds_size(AABB bb)
  {
    return bb.bmax - bb.bmin;
  }

  ONIKA_HOST_DEVICE_FUNC inline double bounds_volume(AABB bb)
  {
    Vec3d v = bb.bmax - bb.bmin;
    return v.x * v.y * v.z;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline AABB block_to_bounds(GridBlock gb, Vec3d origin, double cell_size)
  {
    return AABB{ gb.start*cell_size + origin , gb.end*cell_size + origin };
  }
  
}

