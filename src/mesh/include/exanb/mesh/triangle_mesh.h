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
#include <onika/memory/allocator.h>
#include <onika/math/basic_types.h>
#include <onika/oarray.h>
#include <onika/cuda/ro_shallow_copy.h>
#include <onika/cuda/cuda_math.h>
#include <onika/oarray_stream.h>
#include <onika/cuda/stl_adaptors.h>
#include <exanb/mesh/triangle.h>
#include <exanb/core/grid_algorithm.h>

namespace exanb
{

  using VertexArray = onika::memory::CudaMMVector< onika::math::Vec3d >;
  using TriangleConnectivity = onika::oarray_t<uint64_t,3>;
  using TriangleConnectivityArray = onika::memory::CudaMMVector< TriangleConnectivity >;
  using VertexTriangleCountArray = onika::memory::CudaMMVector<unsigned int>;

  struct TriangleMesh
  {
    VertexArray m_vertices;
    TriangleConnectivityArray m_triangles;

    inline size_t triangle_count() const
    {
      return m_triangles.size();
    }

    inline size_t vertex_count() const
    {
      return m_vertices.size();
    }
    
    inline onika::math::Vec3d vertex(size_t i) const
    {
      return m_vertices[i];
    }

    inline Triangle triangle(size_t i) const
    {
      return { m_vertices[m_triangles[i][0]] , m_vertices[m_triangles[i][1]] , m_vertices[m_triangles[i][2]] };
    }

    inline TriangleConnectivity triangle_connectivity(size_t i) const
    {
      return { m_triangles[i][0] , m_triangles[i][1] , m_triangles[i][2] };
    }

    template<class StreamT>
    inline StreamT& to_stream(StreamT& out)
    {
      size_t nv = 0;
      out << "vertices ("<< vertex_count() <<") :" << std::endl;
      for(const auto& v : m_vertices)
      {
        out << "  "<<(nv++)<<" : "<<v<<std::endl;
      }
      size_t nt = 0;
      out << "triangles ("<< triangle_count() <<") :" << std::endl;
      for(const auto& t : m_triangles)
      {
        out << "  "<<(nt++)<<" : "<< onika::format_array(t) <<std::endl;
      }
      return out;
    }    
  };

  struct TriangleMeshRO
  {
    onika::cuda::ro_shallow_copy_t<VertexArray> m_vertices;
    onika::cuda::ro_shallow_copy_t<TriangleConnectivityArray> m_triangles;
    
    ONIKA_HOST_DEVICE_FUNC
    inline size_t triangle_count() const
    {
      return m_triangles.size();
    }

    ONIKA_HOST_DEVICE_FUNC
    inline size_t vertex_count() const
    {
      return m_vertices.size();
    }
    
    ONIKA_HOST_DEVICE_FUNC
    inline Triangle triangle(size_t i) const
    {
      return { m_vertices[m_triangles[i][0]] , m_vertices[m_triangles[i][1]] , m_vertices[m_triangles[i][2]] };
    }

    ONIKA_HOST_DEVICE_FUNC
    inline TriangleConnectivity triangle_connectivity(size_t i) const
    {
      return { m_triangles[i][0] , m_triangles[i][1] , m_triangles[i][2] };
    }
  };

  inline TriangleMeshRO read_only_view( const TriangleMesh& other )
  {
    return { other.m_vertices , other.m_triangles };
  }

  inline void count_vertex_triangles(const TriangleMesh& trimesh, VertexTriangleCountArray & vtricount)
  {
    const size_t N = trimesh.vertex_count();
    vtricount.assign( N , 0 );
    for(size_t i=0;i<N;i++)
    {
      const auto tricon = trimesh.triangle_connectivity(i);
      for(size_t j=0;j<3;j++) vtricount[tricon[j]] ++ ;
    }
  }
  
  struct GridTriangleIntersectionList
  {
    using CellTriangleArray = onika::memory::CudaMMVector<size_t>;
    onika::math::IJK m_grid_dims;
    double m_cell_size;
    onika::math::Vec3d m_origin;
    CellTriangleArray m_cell_triangles;
  };

  struct GridTriangleIntersectionListRO
  {
    using CellTriangleArray = onika::cuda::ro_shallow_copy_t< onika::memory::CudaMMVector<size_t> >;
    onika::math::IJK m_grid_dims;
    double m_cell_size;
    onika::math::Vec3d m_origin;
    CellTriangleArray m_cell_triangles;
    
    ONIKA_HOST_DEVICE_FUNC
    inline size_t cell_triangle_count(size_t cell_idx) const
    {
      return m_cell_triangles[cell_idx+1] -  m_cell_triangles[cell_idx];
    }

    ONIKA_HOST_DEVICE_FUNC
    inline auto cell_triangles(size_t cell_idx) const
    {
      return onika::cuda::span<const size_t>{ m_cell_triangles.data() + m_cell_triangles[cell_idx] , cell_triangle_count(cell_idx) };
    }

    ONIKA_HOST_DEVICE_FUNC
    inline auto triangles_nearby(const onika::math::Vec3d& r) const
    {
      using namespace onika::math;
      using namespace onika::cuda;
      const auto cell_loc = vclamp( make_ijk( ( ( r - m_origin ) / m_cell_size ) + 0.5 ) , IJK{0,0,0} , m_grid_dims-1 );
      const auto cell_idx = clamp( grid_ijk_to_index( m_grid_dims , cell_loc ) , ssize_t(0) , grid_cell_count(m_grid_dims)-1 );
      return cell_triangles( cell_idx );
    }
  };

  template<class IntegerT = size_t>
  struct IntegerSequenceSpan
  {
    IntegerT m_start;
    IntegerT m_size;
    ONIKA_HOST_DEVICE_FUNC inline IntegerT operator [] (size_t i) const { return m_start+i; }
    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    ONIKA_HOST_DEVICE_FUNC inline auto begin() const { return m_start; }
    ONIKA_HOST_DEVICE_FUNC inline auto end() const { return m_start + m_size; }
  };

  template<class TriangleSpanT = IntegerSequenceSpan<> >
  struct TrivialTriangleLocatorTmpl
  {
    TriangleSpanT m_triangle_index_list;
    ONIKA_HOST_DEVICE_FUNC
    inline const TriangleSpanT& triangles_nearby(const onika::math::Vec3d& r) const
    {
      return m_triangle_index_list;
    }
  };
  using TrivialTriangleLocator = TrivialTriangleLocatorTmpl<>;

  inline GridTriangleIntersectionListRO read_only_view( const GridTriangleIntersectionList& other )
  {
    return { other.m_grid_dims , other.m_cell_size , other.m_origin , other.m_cell_triangles };
  }
}

namespace onika
{
  namespace cuda
  {
    template<> struct ReadOnlyShallowCopyType<exanb::GridTriangleIntersectionList> { using type = exanb::GridTriangleIntersectionListRO; };
    template<> struct ReadOnlyShallowCopyType<exanb::TriangleMesh> { using type = exanb::TriangleMeshRO; };
  }
}

namespace YAML
{ 
  template<>
  struct convert< exanb::TriangleMesh >
  {
    static inline bool decode(const Node& node, exanb::TriangleMesh& v)
    {
      if( ! node.IsMap() ) { return false; }
      v.m_vertices.clear();
      v.m_triangles.clear();
      if( node["vertices"] )
      {
        const auto data = node["vertices"].as< std::vector< onika::math::Vec3d > >();
        v.m_vertices.assign( data.begin() , data.end() );
      }
      if( node["triangles"] )
      {
        const auto triangles = node["triangles"].as< std::vector< std::array<size_t,3> > >();
        for(const auto& t : triangles)
        {
          v.m_triangles.push_back( { t[0] , t[1] , t[2] } );
        }
      }
      return true;
    }
  };

}


