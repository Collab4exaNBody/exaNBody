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

#include <cassert>
#include <iostream>
#include <yaml-cpp/yaml.h>

#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/physics/units.h>
#include <exanb/core/geometry.h>
#include <iomanip>

namespace exanb
{

  class alignas(8) Domain
  {
  public:

    static inline constexpr uint32_t FLAG_PERIODIC_X      = 1u << 0;    
    static inline constexpr uint32_t FLAG_PERIODIC_Y      = 1u << 1;
    static inline constexpr uint32_t FLAG_PERIODIC_Z      = 1u << 2;
    static inline constexpr uint32_t FLAG_MIRROR_X_MIN    = 1u << 3;
    static inline constexpr uint32_t FLAG_MIRROR_X_MAX    = 1u << 4;
    static inline constexpr uint32_t FLAG_MIRROR_Y_MIN    = 1u << 5;
    static inline constexpr uint32_t FLAG_MIRROR_Y_MAX    = 1u << 6;
    static inline constexpr uint32_t FLAG_MIRROR_Z_MIN    = 1u << 7;
    static inline constexpr uint32_t FLAG_MIRROR_Z_MAX    = 1u << 8;
    
    static inline constexpr uint32_t FLAG_EXPANDABLE      = 1u << 16;
    static inline constexpr uint32_t FLAG_XFORM_IDENTITY  = 1u << 17;

    // grid_dimension() and m_grid_dims is the total size (in cells) of the whole simulation domain
    inline IJK grid_dimension() const { return m_grid_dims; }
    inline void set_grid_dimension(const IJK& dims) { m_grid_dims = dims; }
    
    // bounds() is the whole simulation domain physical size
    inline AABB bounds() const { return m_bounds; }
    inline void set_bounds(const AABB& v) { m_bounds = v; }

    inline const Vec3d& origin() const { return m_bounds.bmin; }
    inline const Vec3d& extent() const { return m_bounds.bmax; }
    
    inline Vec3d bounds_size() const { return m_bounds.bmax - m_bounds.bmin; }
    
    // cubic size of cell
    inline double cell_size() const { return m_cell_size; }
    inline void set_cell_size(double v) { m_cell_size=v; }

    // tells if particular directions are periodic or not
    inline bool periodic_boundary_x() const { return get_bit(FLAG_PERIODIC_X); }
    inline void set_periodic_boundary_x(bool b) { set_bit(FLAG_PERIODIC_X,b); if(b) set_bit(FLAG_MIRROR_X_MIN|FLAG_MIRROR_X_MAX,false); }
    inline bool periodic_boundary_y() const { return get_bit(FLAG_PERIODIC_Y); }
    inline void set_periodic_boundary_y(bool b) { set_bit(FLAG_PERIODIC_Y,b); if(b) set_bit(FLAG_MIRROR_Y_MIN|FLAG_MIRROR_Y_MAX,false); }
    inline bool periodic_boundary_z() const { return get_bit(FLAG_PERIODIC_Z); }
    inline void set_periodic_boundary_z(bool b) { set_bit(FLAG_PERIODIC_Z,b); if(b) set_bit(FLAG_MIRROR_Z_MIN|FLAG_MIRROR_Z_MAX,false); }
    inline void set_periodic_boundary(bool x, bool y, bool z) { set_periodic_boundary_x(x); set_periodic_boundary_y(y); set_periodic_boundary_z(z); }

    inline bool mirror_x_min() const { return get_bit(FLAG_MIRROR_X_MIN); }
    inline void set_mirror_x_min(bool b) { set_bit(FLAG_MIRROR_X_MIN,b); if(b) set_periodic_boundary_x(false); }
    inline bool mirror_x_max() const { return get_bit(FLAG_MIRROR_X_MAX); }
    inline void set_mirror_x_max(bool b) { set_bit(FLAG_MIRROR_X_MAX,b); if(b) set_periodic_boundary_x(false); }

    inline bool mirror_y_min() const { return get_bit(FLAG_MIRROR_Y_MIN); }
    inline void set_mirror_y_min(bool b) { set_bit(FLAG_MIRROR_Y_MIN,b); if(b) set_periodic_boundary_y(false); }
    inline bool mirror_y_max() const { return get_bit(FLAG_MIRROR_Y_MAX); }
    inline void set_mirror_y_max(bool b) { set_bit(FLAG_MIRROR_Y_MAX,b); if(b) set_periodic_boundary_y(false); }

    inline bool mirror_z_min() const { return get_bit(FLAG_MIRROR_Z_MIN); }
    inline void set_mirror_z_min(bool b) { set_bit(FLAG_MIRROR_Z_MIN,b); if(b) set_periodic_boundary_z(false); }
    inline bool mirror_z_max() const { return get_bit(FLAG_MIRROR_Z_MAX); }
    inline void set_mirror_z_max(bool b) { set_bit(FLAG_MIRROR_Z_MAX,b); if(b) set_periodic_boundary_z(false); }

    inline bool expandable() const { return get_bit(FLAG_EXPANDABLE); }
    inline void set_expandable(bool b) { set_bit(FLAG_EXPANDABLE,b); }

    inline Mat3d xform() const { return m_xform; }
    inline Mat3d inv_xform() const { return m_inv_xform; }
    inline bool xform_is_identity() const { return get_bit(FLAG_XFORM_IDENTITY); }
    inline double xform_min_scale() const { return m_xform_min_scale; }
    inline double xform_max_scale() const { return m_xform_max_scale; }
    void set_xform(const Mat3d& mat);

  private:
    inline void set_bit(uint32_t mask, bool b)
    {
      if(b) m_flags |=  mask;
      else  m_flags &= ~mask;
    }
    inline bool get_bit(uint32_t mask) const
    {
      return ( m_flags & mask ) == mask;
    }
  
    AABB m_bounds { {0.,0.,0.} , {0.,0.,0.} };
    IJK m_grid_dims { 0, 0, 0 };
    double m_cell_size = 0.0;

    // transformation to the physical space
    Mat3d m_xform = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
    Mat3d m_inv_xform = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
    double m_xform_min_scale = 1.0;
    double m_xform_max_scale = 1.0;

    uint32_t m_flags = FLAG_EXPANDABLE | FLAG_XFORM_IDENTITY;
    uint32_t m_reserved_padding = 0;
  };

  enum class ReadBoundsSelectionMode
  {
    FILE_BOUNDS,
    DOMAIN_BOUNDS,
    COMPUTED_BOUNDS
  };



  // ****************** periodicity computations *******************
  namespace details
  {
    static inline ssize_t pmod(ssize_t i, ssize_t n)
    {
      return ( (i % n) + n) % n ;
    }
  }
  
  inline IJK periodic_location( IJK dims, IJK loc )
  {
    using details::pmod;
    return IJK{ pmod(loc.i,dims.i) , pmod(loc.j,dims.j) , pmod(loc.k,dims.k) };
  }
  
  // compute domain grid location of the cell containing (rx,ry,rz),
  // w.r.t periodic boundary conditions and correct rx,ry and rz accordingly
  // @param domain IN domain description
  // @param r IN/OUT position. may be modified.
  // @return IJK location of surrounding cell in the domain's grid
  inline IJK domain_periodic_location( const Domain& domain, Vec3d& r )
  {
    using details::pmod;
    Vec3d rel_pos = r - domain.origin();
    IJK loc = make_ijk( rel_pos / domain.cell_size() );

    if( ( loc.i<0 || loc.i>=domain.grid_dimension().i ) && domain.periodic_boundary_x() )
    {
      ssize_t old_i = loc.i;
      loc.i = pmod( loc.i , domain.grid_dimension().i );
      r.x += (loc.i - old_i) * domain.cell_size();      
      assert( r.x >= domain.origin().x && r.x < domain.extent().x );
    }

    if( ( loc.j<0 || loc.j>=domain.grid_dimension().j ) && domain.periodic_boundary_y() )
    {
      ssize_t old_j = loc.j;
      loc.j = pmod( loc.j , domain.grid_dimension().j );
      r.y += (loc.j - old_j) * domain.cell_size();
      assert( r.y >= domain.origin().y && r.y < domain.extent().y );
    }

    if( ( loc.k<0 || loc.k>=domain.grid_dimension().k ) && domain.periodic_boundary_z() )
    {
      ssize_t old_k = loc.k;
      loc.k = pmod( loc.k , domain.grid_dimension().k );
      r.z += (loc.k - old_k) * domain.cell_size();
      assert( r.z >= domain.origin().z && r.z < domain.extent().z );
    }

    return loc;
  }

  // find a periodic version of a point that is closest to a reference point
  static inline double adjust_periodic_coordinate( double x, double rx, double d )
  {
    if( std::abs((x+d)-rx) < std::abs(x-rx) )
    {
      do { x += d; } while( std::abs((x+d)-rx) < std::abs(x-rx) );
    }
    else
    {
      while( std::abs((x-d)-rx) < std::abs(x-rx) ) { x -= d; }
    }
    return x;
  }

  static inline Vec3d adjust_periodic_position( Vec3d p, Vec3d rp, Vec3d d )
  {
    return Vec3d { adjust_periodic_coordinate(p.x,rp.x,d.x) , adjust_periodic_coordinate(p.y,rp.y,d.y) , adjust_periodic_coordinate(p.z,rp.z,d.z) };
  }

  inline Vec3d find_periodic_closest_point( Vec3d p, Vec3d rp, AABB b )
  {
    // assert( p.x>=b.bmin.x && p.x<=b.bmax.x && p.y>=b.bmin.y && p.y<=b.bmax.y && p.z>=b.bmin.z && p.z<=b.bmax.z );
    assert( rp.x>=b.bmin.x && rp.x<=b.bmax.x && rp.y>=b.bmin.y && rp.y<=b.bmax.y && rp.z>=b.bmin.z && rp.z<=b.bmax.z );
    return b.bmin + adjust_periodic_position( p-b.bmin , rp-b.bmin , b.bmax - b.bmin ) ;
  }


  // **** utility functions ****

  void compute_domain_bounds(
    Domain& domain,
    ReadBoundsSelectionMode bounds_mode = ReadBoundsSelectionMode::DOMAIN_BOUNDS,
    double enlargement = 0.0,
    const AABB& file_bounds = AABB(),
    const AABB& all_bounds = AABB() ,
    bool pbc_adjust_xform = false
    );

  // check that the domain size match the grid size for directions with periodic boundary conditions
  bool check_domain( const Domain& domain );

  // user input file format output, usefull for output formats without domain information
  template<class StreamT>
  inline StreamT& print_user_input(const Domain& domain, StreamT& out, int indent=0)
  {
    auto space = [indent](unsigned int n) -> std::string { return std::string((indent+n)*2,' '); } ;
    return out << std::setprecision(15)
      << space(0) << "domain:" << std::endl
      << space(1) << "cell_size: " << domain.cell_size() << std::endl
      << space(1) << "bounds: [ ["<<domain.bounds().bmin.x<<","<<domain.bounds().bmin.y<<","<<domain.bounds().bmin.z<<"] , ["<<domain.bounds().bmax.x<<","<<domain.bounds().bmax.y<<","<<domain.bounds().bmax.z<<"] ]"<< std::endl
      << space(1) << "periodic: [ " << std::boolalpha << domain.periodic_boundary_x() << " , " << std::boolalpha << domain.periodic_boundary_y() << " , " << std::boolalpha << domain.periodic_boundary_z() << " ]" << std::endl
      << space(1) << "expandable: " << std::boolalpha << domain.expandable() << std::endl;
  }

  // **** pretty printing ****
  std::ostream& operator << (std::ostream& out, const Domain& domain);
  std::ostream& operator << (std::ostream& out, ReadBoundsSelectionMode x);

} // end of exanb namespace



// **** YAML Conversion ****
namespace YAML
{
  template<> struct convert< exanb::Domain >
  {
    static Node encode(const exanb::Domain& domain);
    static bool decode(const Node& node, exanb::Domain& domain);
  };

  template<> struct convert< exanb::ReadBoundsSelectionMode >
  {
    static Node encode(const exanb::ReadBoundsSelectionMode& v);
    static bool decode(const Node& node, exanb::ReadBoundsSelectionMode& v);
  };
}


