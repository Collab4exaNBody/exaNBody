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
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  template<class ProximityFuncT, bool EnableEdge=true, bool EnableVertex=true>
  struct ParticleMeshProximityFunctor
  {
    GridTriangleIntersectionListRO m_grid_triangles;
    TriangleMeshRO m_triangle_mesh;
    double m_max_dist;
    ProximityFuncT m_func;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t cell_idx, unsigned int p_idx, double rx, double ry, double rz ) const
    {
      const Vec3d r = {rx,ry,rz};
      const auto trigrid_cell_idx = m_grid_triangles.cell_idx_from_coord( r );
      const auto * cell_triangles = m_grid_triangles.cell_triangles_begin( trigrid_cell_idx );
      const int n_triangles =  m_grid_triangles.cell_triangle_count( trigrid_cell_idx );

      // either nearest_v0,nearest_v1 and nearest_v2 are valid (!=-1), in which case there is a valid nearest triangle (with inside==true)
      // either only nearest_v0 and nearest_v1 are valid, in which case there is a valid nearest edge (with inside==true)
      // either only nearest_v0 is valid, in which case there is a nearest vertex
      ssize_t nearest_v0 = -1 , nearest_v1 = -1 , nearest_v2 = -1;
      Vec3d pv;
      double nearest_dist = 0.0;
      double w0=1.0, w1=0.0, w2=0.0;
      bool behind = false;
      
      for(int i=0;i<n_triangles;i++)
      {
        const auto tri_idx = cell_triangles[i];
        const auto v = m_triangle_mesh.triangle_connectivity( tri_idx );
        const Triangle tri = { m_triangle_mesh.m_vertices[v[0]] , m_triangle_mesh.m_vertices[v[1]] , m_triangle_mesh.m_vertices[v[2]] };
        const auto tri_proj = project_triangle_point( tri , r );
        
        // test if nearest triangle is closest element
        if( tri_proj.m_inside && ( nearest_v0 == -1 || abs(tri_proj.m_dist) < nearest_dist ) )
        {
          nearest_v0 = v[0];
          nearest_v1 = v[1];
          nearest_v2 = v[2];
          pv = tri_proj.m_proj;
          nearest_dist = abs(tri_proj.m_dist);
          behind = tri_proj.m_behind;
          w0 = tri_proj.m_u;
          w1 = tri_proj.m_v;
          w2 = tri_proj.m_w;
        }
        else
        {
          // test vertices and edges
          for(int ei=0;ei<3;ei++)
          {
            const int eip1 = (ei+1) % 3 ;
            const auto e0 = tri[ei];

            // test if vertex is nearest element
            if constexpr ( EnableVertex )
            {
              const auto e0_dist = norm( e0 - r );
              if( nearest_v0 == -1 || e0_dist < nearest_dist )
              {
                nearest_v0 = v[ei];
                nearest_v1 = -1;
                nearest_v2 = -1;
                pv = e0;
                nearest_dist = e0_dist;
                w0 = 1.0;
                w1 = 0.0;
                w2 = 0.0;
              }
            }

            // test if edge is nearest element
            if constexpr ( EnableEdge )
            {
              const auto edge_proj = project_edge_point( { e0 , tri[eip1] } , r );
              if( edge_proj.m_inside && ( nearest_v0 == -1 || edge_proj.m_dist < nearest_dist ) )
              {
                nearest_v0 = v[ei];
                nearest_v1 = v[eip1];
                nearest_v2 = -1;
                pv = edge_proj.m_proj;
                nearest_dist = edge_proj.m_dist;
                w0 = edge_proj.m_u;
                w1 = 1 - w0;
                w2 = 0.0;
              }
            }
          }
        }
      }
      
      m_func( r, pv, nearest_dist , cell_idx , p_idx , nearest_v0, nearest_v1, nearest_v2, w0,w1,w2, behind );
      
    }
  };

  template<class FuncT, bool EnableEdge, bool EnableVertex> struct ComputeCellParticlesTraits< ParticleMeshProximityFunctor<FuncT,EnableEdge,EnableVertex> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

}


