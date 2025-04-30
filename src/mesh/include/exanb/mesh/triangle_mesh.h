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
#include <onika/math/basic_types_def.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/oarray.h>
#include <exanb/mesh/triangle.h>
#include <onika/cuda/ro_shallow_copy.h>
#include <onika/oarray_stream.h>

namespace exanb
{

  using VertexAray = onika::memory::CudaMMVector< onika::math::Vec3d >;
  using TriangleConnectivityArray = onika::memory::CudaMMVector< onika::oarray_t<long,3> >;

  template<class _VertexArayT = VertexAray , class _TriangleArrayT = TriangleConnectivityArray >
  struct TriangleMesh
  {
    using VertexArayT = _VertexArayT;
    using TriangleArrayT = _TriangleArrayT;
    VertexArayT m_vertices;
    TriangleArrayT m_triangles;

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

  struct GridTriangleIntersectionList
  {
    onika::math::IJK m_grid_dims;
    double m_cell_size;
    onika::math::AABB m_origin;
    onika::memory::CudaMMVector<size_t> m_cell_triangles;
  };

  struct GridTriangleIntersectionListRO
  {
    onika::math::IJK m_grid_dims;
    double m_cell_size;
    onika::math::Vec3d m_origin; 
    onika::cuda::VectorShallowCopy<const size_t> m_cell_triangles;
    
    GridTriangleIntersectionListRO() = default;
    GridTriangleIntersectionListRO(const GridTriangleIntersectionListRO&) = default;
    GridTriangleIntersectionListRO(GridTriangleIntersectionListRO &&) = default;
    GridTriangleIntersectionListRO& operator = (const GridTriangleIntersectionListRO&) = default;
    GridTriangleIntersectionListRO& operator = (GridTriangleIntersectionListRO &&) = default;

    inline GridTriangleIntersectionListRO(const GridTriangleIntersectionList& other)
      : m_grid_dims(other.m_grid_dims)
      , m_cell_size(other.m_cell_size)
      , m_grid_bounds(other.m_grid_bounds)
      , m_triangle_indices( other.m_triangle_indices.data(), other.m_triangle_indices.size() )
      , m_cell_triangles_start( other.m_cell_triangles_start.data() , other.m_cell_triangles_start.size() )
      , m_cell_triangles_count( other.m_cell_triangles_count.data() , other.m_cell_triangles_start.size() )
      {}

    inline size_t cell_triangle_count(size_t cell_idx)
    {
      return m_cell_triangles[cell_idx+1] -  m_cell_triangles[cell_idx];
    }

    inline const size_t * cell_triangles(size_t cell_idx) const
    {
      return m_cell_triangles.data() + m_cell_triangles[cell_idx];
    }
    
  };

}

namespace onika
{
  namespace cuda
  {
    template<> struct ReadOnlyShallowCopyType< exanb::GridTriangleIntersectionList > { using type = exanb::GridTriangleIntersectionListRO; };
  }
}


namespace YAML
{ 

  template<class VertexArayT , class TriangleArrayT>
  struct convert< exanb::TriangleMesh<VertexArayT,TriangleArrayT> >
  {
    static inline bool decode(const Node& node, exanb::TriangleMesh<VertexArayT,TriangleArrayT>& v)
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
        const auto triangles = node["triangles"].as< std::vector< onika::math::IJK > >();
        for(const auto& t : triangles)
        {
          v.m_triangles.push_back( { t.i , t.j , t.k } );
        }
      }
      return true;
    }
  };

}


