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
#include <exanb/core/domain.h>
#include <exanb/core/log.h>
#include <exanb/core/math_utils.h>
#include <onika/string_utils.h>

namespace exanb
{

  void Domain::set_xform(const Mat3d& mat)
  {
    m_xform = mat;
    set_bit(FLAG_XFORM_IDENTITY,is_identity(m_xform));
    m_xform_min_scale = 1.0;
    m_xform_max_scale = 1.0;
    if( xform_is_identity() )
    {
      m_inv_xform = m_xform;
    }
    else
    {
      m_inv_xform = inverse( m_xform );
      matrix_scale_min_max( m_xform , m_xform_min_scale, m_xform_max_scale );      
      assert( diff_l2_norm( m_xform * m_inv_xform , make_identity_matrix() ) < 1.e-14 );
    }
    
  }


  // **** pretty printing ****
  std::ostream& operator << (std::ostream& out, const Domain& domain)
  {
    const char * sep = ""; auto nsep = [&sep](const std::string& str) -> std::string { auto r=std::string(sep)+str; sep=","; return r; };    
    out << "bounds="<<domain.bounds()
        <<", dims="<<domain.grid_dimension()
        <<", cell_size="<<domain.cell_size()
        <<", periodic="<< std::boolalpha << domain.periodic_boundary_x()<<','<<domain.periodic_boundary_y()<<','<<domain.periodic_boundary_z()
        <<", mirror="<<(domain.mirror_x_min()?nsep("X-"):"")<<(domain.mirror_x_max()?nsep("X+"):"")
                     <<(domain.mirror_y_min()?nsep("Y-"):"")<<(domain.mirror_y_max()?nsep("Y+"):"")
                     <<(domain.mirror_z_min()?nsep("Z-"):"")<<(domain.mirror_z_max()?nsep("Z+"):"")
        <<", xform="<<domain.xform()
        <<", inv="<<domain.inv_xform()
        <<", scale="<<domain.xform_min_scale()<< "/" <<domain.xform_max_scale();
    return out;
  }

  std::ostream& operator << (std::ostream& out, ReadBoundsSelectionMode x)
  {
    switch( x )
    {
      case ReadBoundsSelectionMode::FILE_BOUNDS : out << "FILE"; break;
      case ReadBoundsSelectionMode::DOMAIN_BOUNDS : out << "DOMAIN"; break;
      case ReadBoundsSelectionMode::COMPUTED_BOUNDS : out << "COMPUTED"; break;
    }
    return out;
  }

  // **** utility functions ****

  // check that the domain size match the grid size for directions with periodic boundary conditions
  bool check_domain( const Domain& domain )
  {
    Vec3d domain_grid_size = domain.grid_dimension() * domain.cell_size();
    Vec3d domain_size = domain.bounds_size();
    return ( fabs( 1. - domain_grid_size.x/domain_size.x ) < 1.e-15 )
        && ( fabs( 1. - domain_grid_size.y/domain_size.y ) < 1.e-15 )
        && ( fabs( 1. - domain_grid_size.z/domain_size.z ) < 1.e-15 );
  }


  // adjust domain bounds and/or cell size and/or transform to satisfy periodicity and desired domain bounds
  void compute_domain_bounds(
    Domain& domain,
    ReadBoundsSelectionMode bounds_mode,
    double enlarge_bounds,
    const AABB& file_bounds,
    const AABB& all_bounds,
    bool pbc_adjust_xform )
  {
    static constexpr double dom_err_epsilon = 1.e-12;
    // auto & ldbg = lout;

    Vec3d dom_size;
    switch( bounds_mode )
    {
      case ReadBoundsSelectionMode::FILE_BOUNDS : // bounds stored in data file
        dom_size = bounds_size( file_bounds );
        break;
      case ReadBoundsSelectionMode::COMPUTED_BOUNDS : // bounds of all particles read from file
        dom_size = bounds_size( all_bounds );
        break;
      case ReadBoundsSelectionMode::DOMAIN_BOUNDS : // from YAML config file domain: { bounds: ... }
        dom_size = bounds_size( domain.bounds() );
        break;
    }

    // compute domain size and enlarge it as requested
    Vec3d enlargement { enlarge_bounds , enlarge_bounds, enlarge_bounds };
    dom_size = dom_size + enlargement * 2.;

    // std::cout << "dom_size:" << dom_size << std::endl;

    IJK dims = domain.grid_dimension();
    double cell_size = domain.cell_size();

    // std::cout << "cell_size:" << cell_size << std::endl;
    if( dims == IJK{0,0,0} )
    {
      if( cell_size == 0. )
      {
        lerr<<"Warning: you must define either grid size or cell size. Cell size forced to 10.0"<<std::endl;
        cell_size = 10.0;
      }
      dims = make_ijk( dom_size / cell_size );
      dims.i = std::max( static_cast<ssize_t>(1) , dims.i );
      dims.j = std::max( static_cast<ssize_t>(1) , dims.j );
      dims.k = std::max( static_cast<ssize_t>(1) , dims.k );
    }
    domain.set_grid_dimension( dims );

    // std::cout << "OK1; domain.m_grid_dims:" << domain.m_grid_dims << std::endl;

    ldbg << "grid dims set to "<< domain.grid_dimension() << std::endl;

    // set domain bounds to file or recomputed bounds, plus given elargement
    switch( bounds_mode )
    {
      case ReadBoundsSelectionMode::FILE_BOUNDS :
        domain.set_bounds( file_bounds );
        break;
      case ReadBoundsSelectionMode::COMPUTED_BOUNDS :
        domain.set_bounds( all_bounds );
        break;
      case ReadBoundsSelectionMode::DOMAIN_BOUNDS :
        // keep value passed in domain.m_bounds
        break;
    }

    // apply enlargment
    domain.set_bounds( AABB{ domain.bounds().bmin-enlargement , domain.bounds().bmax+enlargement } );

    // ==== at this point, domain bounds are the real ones defined by user ====
    // cell_size is the one prefered by user

    // update domain size from the final bounds
    dom_size = bounds_size( domain.bounds() );
    ldbg << "requested bounds = "<< domain.bounds() << std::endl;
    
    // from now cell_size is the definitive cell size
    Vec3d cell_shape = dom_size / domain.grid_dimension();

    // cells have to be cubic
    if( !pbc_adjust_xform )
    {
      ldbg << "cell_size : "<< cell_size << " -> ";
      cell_size = std::min( std::min( cell_shape.x , cell_shape.y ) , cell_shape.z );
      ldbg << cell_size << std::endl;
    }
    
    assert( cell_size > 0. );
    domain.set_cell_size( cell_size );
    
    Vec3d domain_grid_size = domain.grid_dimension() * domain.cell_size();
    
    const double bound_max_length = std::max( std::max( dom_size.x , dom_size.y ) , dom_size.z );
    const double grid_bound_err = std::max( std::max( std::fabs(domain_grid_size.x-dom_size.x) , std::fabs(domain_grid_size.y-dom_size.y) ) , std::fabs(domain_grid_size.z-dom_size.z) ) / bound_max_length;
    
    ldbg << "domain grid size / domain bounds err = "<< grid_bound_err << std::endl;
    if( grid_bound_err > 1.e-15 )
    {
      domain.set_grid_dimension( { static_cast<ssize_t>( std::ceil( dom_size.x / cell_size ) )
                                 , static_cast<ssize_t>( std::ceil( dom_size.y / cell_size ) )
                                 , static_cast<ssize_t>( std::ceil( dom_size.z / cell_size ) ) } );
      ldbg << "adapted grid dims = "<< domain.grid_dimension() << std::endl;
    }

    // for non periodic axis, take care that no particles are left on the upper boundary. add an extra cell layer if needed
    {
      IJK d = domain.grid_dimension();
      if( !domain.periodic_boundary_x() && domain.bounds().bmax.x == all_bounds.bmax.x )
      {
        ldbg << "INFO: adjust domain grid dimension (+1) along X axis" << std::endl;
        ++ d.i;
      }
      if( !domain.periodic_boundary_y() && domain.bounds().bmax.y == all_bounds.bmax.y )
      {
        ldbg << "INFO: adjust domain grid dimension (+1) along Y axis" << std::endl;
        ++ d.j;
      }
      if( !domain.periodic_boundary_z() && domain.bounds().bmax.z == all_bounds.bmax.z )
      {
        ldbg << "INFO: adjust domain grid dimension (+1) along Z axis" << std::endl;
        ++ d.k;
      }
      domain.set_grid_dimension( d );
    }

    // now, grid can be slightly larger than the domain bounds.
    // this is ok until we have boundary conditions
    domain_grid_size = domain.grid_dimension() * domain.cell_size();
    bool domain_bounds_adjusted = false;
    
    if( pbc_adjust_xform )
    {
      ldbg << "pbc_adjust_xform: cell_size = "<< domain.cell_size() << std::endl;
    
      IJK grid_dims = domain.grid_dimension();

      if( domain.periodic_boundary_x() && ( dom_size.x / domain_grid_size.x ) < 1.0 )
      {
        ldbg << "X axis : prefer smaller cell / bigger scaling, size : "<<grid_dims.i <<" -> "<<grid_dims.i-1<<std::endl;
        -- grid_dims.i;
        assert( grid_dims.i >= 1 );
      }
      if( domain.periodic_boundary_y() && ( dom_size.y / domain_grid_size.y ) < 1.0 )
      {
        ldbg << "Y axis : prefer smaller cell / bigger scaling, size : "<<grid_dims.j <<" -> "<<grid_dims.j-1<<std::endl;
        -- grid_dims.j;
        assert( grid_dims.j >= 1 );
      }
      if( domain.periodic_boundary_z() && ( dom_size.z / domain_grid_size.z ) < 1.0 )
      {
        ldbg << "Z axis : prefer smaller cell / bigger scaling, size : "<<grid_dims.k <<" -> "<<grid_dims.k-1<<std::endl;
        -- grid_dims.k;
        assert( grid_dims.k >= 1 );
      }
      domain.set_grid_dimension( grid_dims );
      domain_grid_size = domain.grid_dimension() * domain.cell_size();
    
      double dom_scale_x = dom_size.x / domain_grid_size.x;
      double dom_scale_y = dom_size.y / domain_grid_size.y;
      double dom_scale_z = dom_size.z / domain_grid_size.z;
      
      double dom_err_x = std::fabs( dom_scale_x - 1.0 );
      double dom_err_y = std::fabs( dom_scale_y - 1.0 );
      double dom_err_z = std::fabs( dom_scale_z - 1.0 );
      
      Vec3d domain_scaling { 1.0, 1.0, 1.0 };

      if( domain.periodic_boundary_x() && dom_err_x > dom_err_epsilon )
      {
        ldbg << "INFO: scaling X for periodicity requirement: size "<<dom_size.x<<" -> "<<domain_grid_size.x<<" , scaling "<<domain_scaling.x<<" -> "<<dom_scale_x << std::endl;
        dom_size.x = domain_grid_size.x;
        domain_scaling.x = dom_scale_x;
        domain_bounds_adjusted = true;
      }
      if( domain.periodic_boundary_y() && dom_err_y > dom_err_epsilon )
      {
        ldbg << "INFO: scaling Y for periodicity requirement: size "<<dom_size.y<<" -> "<<domain_grid_size.y<<" , scaling "<<domain_scaling.y<<" -> "<<dom_scale_y << std::endl;
        dom_size.y = domain_grid_size.y;
        domain_scaling.y = dom_scale_y;
        domain_bounds_adjusted = true;
      }
      if( domain.periodic_boundary_z() && dom_err_z > dom_err_epsilon )
      {
        ldbg << "INFO: scaling Z for periodicity requirement: size "<<dom_size.z<<" -> "<<domain_grid_size.z<<" , scaling "<<domain_scaling.z<<" -> "<<dom_scale_z << std::endl;
        dom_size.z = domain_grid_size.z;
        domain_scaling.z = dom_scale_z;
        domain_bounds_adjusted = true;
      }
      if( domain_bounds_adjusted )
      {
        domain.set_xform( diag_matrix(domain_scaling) );
        Vec3d dmin = domain.bounds().bmin;
        ldbg << "scaled domain bounds = ("<<dmin<<") - ("<<dmin+ (dom_size*domain_scaling) << ")" << std::endl;
        if( ! is_diagonal( domain.xform() ) )
        {
          lerr << "Adjusted bounds cannot work because domain tranform is not diagonal (not a pure scaling matrix)\n";
          std::abort();
        }
        auto scaled_bounds = domain.bounds();
        scaled_bounds.bmin = domain.inv_xform() * scaled_bounds.bmin;
        scaled_bounds.bmax = domain.inv_xform() * scaled_bounds.bmax;
        domain.set_bounds( scaled_bounds );
      }
    }
    else
    {
      double dom_scale_x = dom_size.x / domain_grid_size.x;
      double dom_scale_y = dom_size.y / domain_grid_size.y;
      double dom_scale_z = dom_size.z / domain_grid_size.z;
      
      double dom_err_x = std::fabs( dom_scale_x - 1.0 );
      double dom_err_y = std::fabs( dom_scale_y - 1.0 );
      double dom_err_z = std::fabs( dom_scale_z - 1.0 );

      if( domain.periodic_boundary_x() && dom_err_x > dom_err_epsilon )
      {
        lerr << "WARNING: domain bounds adjusted along X axis to match periodicity requirement (err="<<dom_err_x<<")" << std::endl;
        dom_size.x = domain_grid_size.x;
        domain_bounds_adjusted = true;
      }
      if( domain.periodic_boundary_y() && dom_err_y > dom_err_epsilon )
      {
        lerr << "WARNING: domain bounds adjusted along Y axis to match periodicity requirement (err="<<dom_err_y<<")" << std::endl;
        dom_size.y = domain_grid_size.y;
        domain_bounds_adjusted = true;
      }
      if( domain.periodic_boundary_z() && dom_err_z > dom_err_epsilon )
      {
        lerr << "WARNING: domain bounds adjusted along Z axis to match periodicity requirement (err="<<dom_err_z<<")" << std::endl;
        dom_size.z = domain_grid_size.z;
        domain_bounds_adjusted = true;
      }
      if( domain_bounds_adjusted )
      {
        auto bmin = domain.bounds().bmin;
        domain.set_bounds( AABB{ bmin , bmin + dom_size } );
      }
    }
    
    if( domain_bounds_adjusted )
    {
      ldbg << "adjusted bounds = "<<domain.bounds() << std::endl;
      ldbg << "adjusted xform  = "<<domain.xform() << std::endl;
    }

    assert( check_domain(domain) );
  }


} // end of exanb namespace

// **** YAML Conversion ****
namespace YAML
{    
  Node convert< exanb::Domain >::encode(const exanb::Domain& domain)
  {
    Node node;
    node["bounds"] = domain.bounds();
    node["grid_dims"] = domain.grid_dimension();
    node["cell_size"]["value"] = domain.cell_size();
    node["cell_size"]["unity"] = "ang";
    std::vector<bool> p = { domain.periodic_boundary_x(), domain.periodic_boundary_y(), domain.periodic_boundary_z() };
    node["periodic"] = p;
    node["expandable"] = domain.expandable();
    std::vector<bool> m = { domain.mirror_x_min(),domain.mirror_x_max(), domain.mirror_y_min(),domain.mirror_y_max(), domain.mirror_z_min(),domain.mirror_z_max() };
    node["mirror"] = m;
    return node;
  }

  bool convert< exanb::Domain >::decode(const Node& node, exanb::Domain& domain)
  {
    if( ! node.IsMap() ) { return false; }

    domain = exanb::Domain();

    if(node["bounds"])
    {
      domain.set_bounds( node["bounds"].as<AABB>() );
    }
    if(node["grid_dims"])
    {
      domain.set_grid_dimension( node["grid_dims"].as<IJK>() );
    }
    if(node["cell_size"])
    {
      domain.set_cell_size( node["cell_size"].as<Quantity>().convert() );
    }
    if(node["periodic"])
    {
      if( node["periodic"].size() != 3 ) { return false; }
      domain.set_periodic_boundary_x( node["periodic"][0].as<bool>() );
      domain.set_periodic_boundary_y( node["periodic"][1].as<bool>() );
      domain.set_periodic_boundary_z( node["periodic"][2].as<bool>() );
    }
    if(node["mirror"])
    {
      if( ! node["mirror"].IsSequence() ) { return false; }
      for(auto m : node["mirror"])
      {
        if( exanb::str_tolower(m.as<std::string>()) == "x-" ) { domain.set_mirror_x_min(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "x+" ) { domain.set_mirror_x_max(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "x" )  { domain.set_mirror_x_min(true); domain.set_mirror_x_max(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "y-" ) { domain.set_mirror_y_min(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "y+" ) { domain.set_mirror_y_max(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "y" )  { domain.set_mirror_y_min(true); domain.set_mirror_y_max(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "z-" ) { domain.set_mirror_z_min(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "z+" ) { domain.set_mirror_z_max(true); }
        if( exanb::str_tolower(m.as<std::string>()) == "z" )  { domain.set_mirror_z_min(true); domain.set_mirror_z_max(true); }
      }
    }
    if(node["expandable"])
    {
      domain.set_expandable( node["expandable"].as<bool>() );
    }
    if(node["xform"])
    {
      domain.set_xform( node["xform"].as<Mat3d>() );
    }
    return true;
  }
    
  Node convert< exanb::ReadBoundsSelectionMode >::encode(const exanb::ReadBoundsSelectionMode& v)
  {
    Node node;
    switch( v )
    {
      case exanb::ReadBoundsSelectionMode::FILE_BOUNDS : node = "FILE"; break;
      case exanb::ReadBoundsSelectionMode::DOMAIN_BOUNDS : node = "DOMAIN"; break;
      case exanb::ReadBoundsSelectionMode::COMPUTED_BOUNDS : node = "COMPUTED"; break;
    }
    return node;
  }
  
  bool convert< exanb::ReadBoundsSelectionMode >::decode(const Node& node, exanb::ReadBoundsSelectionMode& v)
  {
    if( node.as<std::string>() == "FILE" ) { v = exanb::ReadBoundsSelectionMode::FILE_BOUNDS; return true; }
    if( node.as<std::string>() == "COMPUTED" ) { v = exanb::ReadBoundsSelectionMode::COMPUTED_BOUNDS; return true; }
    if( node.as<std::string>() == "DOMAIN" ) { v = exanb::ReadBoundsSelectionMode::DOMAIN_BOUNDS; return true; }
    return false;
  }


}



