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
#include <onika/cuda/cuda_math.h>
#include <onika/oarray_stream.h>

#include <span>

namespace exanb
{

  using VertexAray = onika::memory::CudaMMVector< onika::math::Vec3d >;
  using TriangleConnectivity = onika::oarray_t<uint64_t,3>;
  using TriangleConnectivityArray = onika::memory::CudaMMVector< TriangleConnectivity >;
  using VertexTriangleCountArray = onika::memory::CudaMMVector<unsigned int>;

  template<class _VertexArayT = VertexAray , class _TriangleArrayT = TriangleConnectivityArray >
  struct TriangleMeshTmpl
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

  using TriangleMesh = TriangleMeshTmpl<>;
  using TriangleMeshRO = TriangleMeshTmpl< onika::cuda::ro_shallow_copy_t<TriangleMesh::VertexArayT> , onika::cuda::ro_shallow_copy_t<TriangleMesh::TriangleArrayT> >;

  inline TriangleMeshRO read_only_view( const TriangleMesh& other )
  {
    return { other.m_vertices , other.m_triangles };
  }

  template<class VertexArayT , class TriangleArrayT >
  inline void count_vertex_triangles(const TriangleMeshTmpl<VertexArayT,TriangleArrayT>& trimesh, VertexTriangleCountArray & vtricount)
  {
    const size_t N = trimesh.vertex_count();
    vtricount.assign( N , 0 );
    for(size_t i=0;i<N;i++)
    {
      const auto tricon = trimesh.triangle_connectivity(i);
      for(size_t j=0;j<3;j++) vtricount[tricon[j]] ++ ;
    }
  }
  
  template<class _CellTriangleArrayT = onika::memory::CudaMMVector<size_t> >
  struct GridTriangleIntersectionListTmpl
  {
    using CellTriangleArray = _CellTriangleArrayT;
    onika::math::IJK m_grid_dims;
    double m_cell_size;
    onika::math::Vec3d m_origin;
    CellTriangleArray m_cell_triangles;
    
    ONIKA_HOST_DEVICE_FUNC
    inline size_t cell_idx_from_coord(const onika::math::Vec3d& r) const
    {
      using namespace onika::math;
      using namespace onika::cuda;
      const auto cell_loc = vclamp( make_ijk( ( ( r - m_origin ) / m_cell_size ) + 0.5 ) , IJK{0,0,0} , m_grid_dims-1 );
      return clamp( grid_ijk_to_index( m_grid_dims , cell_loc ) , ssize_t(0) , grid_cell_count(m_grid_dims)-1 );
    }

    ONIKA_HOST_DEVICE_FUNC
    inline size_t cell_triangle_count(size_t cell_idx) const
    {
      return m_cell_triangles[cell_idx+1] -  m_cell_triangles[cell_idx];
    }

    ONIKA_HOST_DEVICE_FUNC
    inline const size_t * cell_triangles_begin(size_t cell_idx) const
    {
      return m_cell_triangles.data() + m_cell_triangles[cell_idx];
    }

    ONIKA_HOST_DEVICE_FUNC
    inline auto cell_triangles(size_t cell_idx) const
    {
      return std::span<const size_t>( cell_triangles_begin(cell_idx) , cell_triangle_count(cell_idx) );
    }
  };
  
  using GridTriangleIntersectionList = GridTriangleIntersectionListTmpl<>;
  using GridTriangleIntersectionListRO = GridTriangleIntersectionListTmpl< onika::cuda::ro_shallow_copy_t<GridTriangleIntersectionList::CellTriangleArray> >;
  
  inline GridTriangleIntersectionListRO read_only_view( const GridTriangleIntersectionList& other )
  {
    return { other.m_grid_dims , other.m_cell_size , other.m_origin , other.m_cell_triangles };
  }
}

namespace onika
{
  namespace cuda
  {
    template<class CellTriangleArrayT> struct ReadOnlyShallowCopyType< exanb::GridTriangleIntersectionListTmpl<CellTriangleArrayT> > { using type = exanb::GridTriangleIntersectionListTmpl< onika::cuda::ro_shallow_copy_t<CellTriangleArrayT> >; };
    template<class VertexArayT , class TriangleArrayT> struct ReadOnlyShallowCopyType< exanb::TriangleMeshTmpl<VertexArayT,TriangleArrayT> > { using type = exanb::TriangleMeshTmpl< onika::cuda::ro_shallow_copy_t<VertexArayT> , onika::cuda::ro_shallow_copy_t<TriangleArrayT> >; };
  }
}

namespace YAML
{ 

  template<class VertexArayT , class TriangleArrayT>
  struct convert< exanb::TriangleMeshTmpl<VertexArayT,TriangleArrayT> >
  {
    static inline bool decode(const Node& node, exanb::TriangleMeshTmpl<VertexArayT,TriangleArrayT>& v)
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


