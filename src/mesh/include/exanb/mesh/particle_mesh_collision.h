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

#include <exanb/mesh/triangle_mesh.h>
#include <exanb/mesh/triangle_math.h>
#include <exanb/mesh/edge_math.h>
#include <onika/math/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  template<class GridTriangleLocator, class CollisionFuncT>
  struct ParticleMeshCollisionFunctor
  {
    GridTriangleLocator m_grid_triangles;
    TriangleMeshRO m_triangle_mesh;
    double m_delta_t;
    CollisionFuncT m_func;
    
    ONIKA_HOST_DEVICE_FUNC
    inline void operator () ( size_t cell_idx, unsigned int p_idx, double rx, double ry, double rz, double vx, double vy, double vz) const
    {
      const Vec3d r0 = {rx,ry,rz};
      const Vec3d v = {vx,vy,vz};
      const Vec3d r1 = r0 + v * m_delta_t;
      const auto triangles_list = m_grid_triangles.triangles_nearby( r0 );
      const int n_triangles = triangles_list.size(); //m_grid_triangles.cell_triangle_count( trigrid_cell_idx );

      assert( ! m_grid_triangles.exceeds_maximum_distance( norm(v*m_delta_t) ) );

      ssize_t nearest_triangle = -1;
      double edge_u = 0.0;
      double tri_w0 = 0.0;
      double tri_w1 = 0.0;
      double tri_w2 = 0.0;

      for(int i=0;i<n_triangles;i++)
      {
        const auto tri_idx = triangles_list[i];
        const auto v = m_triangle_mesh.triangle_connectivity( tri_idx );
        const Triangle tri = { m_triangle_mesh.m_vertices[v[0]] , m_triangle_mesh.m_vertices[v[1]] , m_triangle_mesh.m_vertices[v[2]] };
        const auto tri_edge = triangle_edge_intersection( tri , { r0 , r1 } );
        
        // test if nearest triangle is closest element
        if( tri_edge.m_intersect && ( nearest_triangle == -1 || tri_edge.m_edge_u < edge_u ) )
        {
          nearest_triangle = tri_idx;
          edge_u = tri_edge.m_edge_u;
          tri_w0 = tri_edge.m_tri_u;
          tri_w1 = tri_edge.m_tri_v;
          tri_w2 = tri_edge.m_tri_w;
        }
      }
      
      if( nearest_triangle != -1 ) m_func( cell_idx , p_idx, nearest_triangle, r0, v, edge_u * m_delta_t , tri_w0, tri_w1, tri_w2 );
    }
  };

  template<class FuncT> using GridParticleTriangleCollision = ParticleMeshCollisionFunctor<GridTriangleIntersectionListRO,FuncT>;
  template<class FuncT> using ParticleTriangleCollision = ParticleMeshCollisionFunctor<TrivialTriangleLocator,FuncT>;

  template<class LocatorT, class FuncT> struct ComputeCellParticlesTraits< ParticleMeshCollisionFunctor<LocatorT,FuncT> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

}


