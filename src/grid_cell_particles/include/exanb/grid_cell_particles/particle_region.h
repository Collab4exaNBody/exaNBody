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

#include <onika/math/basic_types_def.h>
#include <onika/math/matrix4d.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <onika/cuda/ro_shallow_copy.h>
#include <onika/memory/allocator.h>

#include <yaml-cpp/yaml.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/yaml/yaml_enum.h>
#include <onika/log.h>
#include <exanb/core/geometry.h>

#include <limits>
#include <memory>

//EXANB_YAML_ENUM(exanb,ParticleRegionIdInitMode,USER_IDS,GEOM_UNION,GEOM_INTERSECT,GEOM_SELF);

namespace exanb
{  
  struct ParticleRegion
  {
    static constexpr size_t MAX_NAME_LEN = 13;
    static constexpr AABB NO_BOUNDS = { { -onika::cuda::numeric_limits<double>::infinity , -onika::cuda::numeric_limits<double>::infinity , -onika::cuda::numeric_limits<double>::infinity } , 
                      {  onika::cuda::numeric_limits<double>::infinity ,  onika::cuda::numeric_limits<double>::infinity ,  onika::cuda::numeric_limits<double>::infinity } };
    
    // default id range includes all possible ids
    uint64_t m_id_start = 0;
    uint64_t m_id_end = std::numeric_limits<uint64_t>::max();
    
    // default bounds includes the whole 3D space
    AABB m_bounds = NO_BOUNDS;
                      
    // default quadric is full 0, so that all points in space are detected inside quadric
    Mat4d m_quadric = { { {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} } };
    
    bool m_id_range_flag = false;
    bool m_bounds_flag   = false;
    bool m_quadric_flag  = false;
    char m_name[MAX_NAME_LEN] = { '\0' };
    
    ONIKA_HOST_DEVICE_FUNC inline void transform_quadric( const Mat4d& M )
    {
      const auto M_inv = inverse(M);
      m_quadric = transpose(M_inv) * m_quadric * M_inv;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline bool contains( const Vec3d& r , uint64_t id ) const
    {
      return id>=m_id_start && id<m_id_end && is_inside(m_bounds,r) && quadric_eval(m_quadric,r)<=0.0 ;
    }
    
    inline void set_name(const std::string& s)
    {
      std::strncpy( m_name , s.c_str() , MAX_NAME_LEN-1 );
      m_name[ MAX_NAME_LEN - 1 ] = '\0';
      if( s.size() >= MAX_NAME_LEN )
      {
        lerr << "Warning: region name '"<<s<<"' truncated to '"<<name()<<"'"<<std::endl;
      }
    }
    
    ONIKA_HOST_DEVICE_FUNC inline const char* name() const { return m_name; }
    
    // includes all points in space
    ONIKA_HOST_DEVICE_FUNC inline void reset_geometry()
    {
      m_bounds = NO_BOUNDS; 
      m_quadric = Mat4d{ { {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} } };     
    }

    ONIKA_HOST_DEVICE_FUNC inline void reset_id_range()
    {
      m_id_start = 0;
      m_id_end = onika::cuda::numeric_limits<uint64_t>::max;      
    }

  };

  using ParticleRegions = onika::memory::CudaMMVector<ParticleRegion>;
  
  struct ParticleRegionCSG
  {
    static constexpr unsigned int MAX_REGION_OPERANDS = 32; // maximum operands in expression binary tree, including constants
    ParticleRegions m_regions;
    uint64_t m_expr = 0;
    unsigned int m_nb_operands = 0;
    unsigned int m_operand_places[32];
    std::string m_user_expr;
    void build_from_expression_string(const ParticleRegion* regions , size_t nb_regions);
  };

  struct ParticleRegionCSGShallowCopy
  {
    const ParticleRegion * __restrict__ m_regions = nullptr;
    uint64_t m_expr = 0;
    uint8_t m_nb_regions = 0;
    uint8_t m_nb_operands = 0;
    uint8_t m_nb_operands_log2 = 0;
    uint8_t m_operand_places[61];

    ParticleRegionCSGShallowCopy() = default;
    ParticleRegionCSGShallowCopy(const ParticleRegionCSGShallowCopy&) = default;
    ParticleRegionCSGShallowCopy(ParticleRegionCSGShallowCopy&&) = default;
    ParticleRegionCSGShallowCopy(const ParticleRegionCSG& prcsg) { init_from(prcsg); }

    ParticleRegionCSGShallowCopy& operator = (const ParticleRegionCSGShallowCopy&) = default;
    ParticleRegionCSGShallowCopy& operator = (ParticleRegionCSGShallowCopy&&) = default;
    ParticleRegionCSGShallowCopy& operator = (const ParticleRegionCSG& prcsg) { init_from(prcsg); return *this; }

    inline void init_from(const ParticleRegionCSG& prcsg)
    {
      m_regions = prcsg.m_regions.data();
      m_expr = prcsg.m_expr;
      m_nb_regions = prcsg.m_regions.size();
      m_nb_operands = prcsg.m_nb_operands;
      unsigned int n = static_cast<unsigned int>(m_nb_operands) ; // keep an empty operand, wich can be used for final negation
      for(unsigned int i=0;i<n;i++) m_operand_places[i] = prcsg.m_operand_places[i];
      m_nb_operands_log2 = 0;
      while( n > 1 ) { ++m_nb_operands_log2 ; n=n>>1; }
    }
    
    ONIKA_HOST_DEVICE_FUNC inline bool contains(const Vec3d& r , uint64_t id) const
    {
      uint64_t rvalues = 0;
//      lout << "regions =";
      for(int i=0;i<m_nb_regions;i++)
      {
        uint64_t C = m_regions[i].contains(r,id);
//        lout << " "<<C;
        rvalues |= ( C << i );
      }
//      lout<<std::endl;

//      lout << "operands =";
      // when object contains no expression at all, it implicitly returns true;
      uint64_t values = ( m_nb_operands > 0 ) ? 0 : 1;
      for(unsigned int i=0;i<m_nb_operands;i++)
      {
//        lout << " " << (int)m_operand_places[i];
        values |= ( ( rvalues >> m_operand_places[i] ) & 1ull ) << i;
      }
//      lout<<std::endl;
      
      uint64_t expr = m_expr;
      unsigned int nop = m_nb_operands;
      for(unsigned int i=0;i<m_nb_operands_log2;i++)
      {
//        lout << "Evaluate round "<<i<<std::endl
//             << "\tvalues =";
//        for(unsigned int i=0;i<nop;i++) lout << " " << ( (values>>i) & 1ull );
//        lout<<std::endl<<"\texpr   =";
//        for(unsigned int i=0;i<nop;i++) lout << " " << ( (expr>>i) & 1ull );
//        lout<<std::endl;

        values = values ^ expr; // apply input optional negation
//        lout << "\txor    =";
//        for(unsigned int i=0;i<nop;i++) lout << " " << ( (values>>i) & 1ull );
//        lout<<std::endl;
        
        values = values & (values>>1); // execute AND operation
//        lout << "\tand    =";
//        for(unsigned int i=0;i<nop;i++)
//        {
//          if(i%2==0) lout << " " << ( (values>>i) & 1ull );
//          else lout << " x";
//        }
//        lout<<std::endl;

        // prepare next operations for next round
        expr = expr >> nop;
        nop = nop / 2;

        // compact operation results
        uint64_t next = 0;
        for(unsigned int j=0;j<nop;j++) 
        {
          next |= ( ( ( values >> (j*2) ) & 1ull ) << j );
        }
        values = next;
//        lout << "\tpacked =";
//        for(unsigned int i=0;i<nop;i++) lout << " " << ( (values>>i) & 1ull );
//        lout<<std::endl;
      }
//      lout << "result = "<< (values&1ull) <<std::endl;
      return ( values & 1ull ) != 0;
    }
  };

}

namespace onika
{
  namespace cuda
  {
    template<> struct ReadOnlyShallowCopyType< exanb::ParticleRegionCSG > { using type = exanb::ParticleRegionCSGShallowCopy; };
  }
}

namespace YAML
{

  template<> struct convert< exanb::ParticleRegionCSG >
  {
    static inline bool decode(const Node& node, exanb::ParticleRegionCSG & prcsg)
    {
      if( ! node.IsScalar() )
      {
        exanb::fatal_error() << "region must be a string" << std::endl;
        return false;
      }
      prcsg = exanb::ParticleRegionCSG{};
      prcsg.m_user_expr = node.as<std::string>();
      return true;
    }
  };


  template<> struct convert< exanb::ParticleRegion >
  {
    static inline Node encode(const exanb::ParticleRegion& r)
    {
      Node node;
      node["id_range"] = std::vector<uint64_t> { r.m_id_start , r.m_id_end };
      node["bounds"] = r.m_bounds;
      node["quadric"] = r.m_quadric;
      node["name"] = std::string( r.name() );
      return node;
    }
    static inline bool decode(const Node& named_region, exanb::ParticleRegion & R)
    {
      static constexpr exanb::ParticleRegion empty_region={};

      R = exanb::ParticleRegion{};
      if( ! named_region.IsMap() )
      {
        exanb::lerr << "Region defintion should be definaed as a map of map 'name: { ... }' " << std::endl;
        exanb::lerr<<"Yaml='"; onika::yaml::dump_node_to_stream( exanb::lerr , named_region ); exanb::lerr <<"'"<<std::endl;
        return false;
      }
      
      std::string name = named_region.begin()->first.as<std::string>();
      Node node = named_region.begin()->second;
      if( ! node.IsMap() )
      {
        exanb::lerr << "Region must be defined with a map" << std::endl;
        exanb::lerr<<"Yaml='"; onika::yaml::dump_node_to_stream( exanb::lerr , node ); exanb::lerr <<"'"<<std::endl;
        return false;
      }

      R.set_name( name );
      //std::strncpy( R.m_name , name.c_str() , exanb::ParticleRegion::MAX_NAME_LEN-1 );
      //R.m_name[ exanb::ParticleRegion::MAX_NAME_LEN - 1 ] = '\0';
      
      if( node["id_range"] )
      {
        if( ! node["id_range"].IsSequence() )
        {
          exanb::lerr << "id_range must be a sequence" << std::endl;
          exanb::lerr<<"Yaml='"; onika::yaml::dump_node_to_stream( exanb::lerr , node ); exanb::lerr <<"'"<<std::endl;
          return false;
        }
        const auto idrange = node["id_range"].as<std::vector<uint64_t> >();
        if( idrange.size() != 2 ) { exanb::lerr << "Bad id range format" << std::endl; return false; }
        R.m_id_start = idrange[0];
        R.m_id_end = idrange[1];
      }
      if( node["bounds"] ) R.m_bounds = node["bounds"].as<exanb::AABB>();
      if( node["quadric"] )
      {
        if( ! node["quadric"]["shape"] ) { exanb::lerr << "Quadric has no shape" << std::endl; return false; }
        R.m_quadric = node["quadric"]["shape"].as<exanb::Mat4d>();
        if( node["quadric"]["transform"] )
        {
          R.transform_quadric( node["quadric"]["transform"].as<exanb::Mat4d>() );
        }
      }
      
      R.m_id_range_flag = ( R.m_id_start!=empty_region.m_id_start || R.m_id_end!=empty_region.m_id_end );
      R.m_bounds_flag = ( R.m_bounds != empty_region.m_bounds );
      R.m_quadric_flag = ! is_zero(R.m_quadric);
      return true;
    }
  };

  // ParticleRegions is an std::vector, but some compilers needs some specialization help
  template<> struct convert< exanb::ParticleRegions >
  {
    static inline bool decode(const Node& node, exanb::ParticleRegions & regions)
    {
      auto v = node.as< std::vector<exanb::ParticleRegion> >();
      regions.assign( v.begin() , v.end() );
      return true;
    }
  };

  
}


