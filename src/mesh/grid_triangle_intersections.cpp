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

#include <onika/scg/operator.h>
#include <exanb/core/domain.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>

#include <exanb/mesh/triangle_mesh.h>
#include <exanb/mesh/triangle_math.h>
#include <exanb/mesh/edge_math.h>
#include <onika/math/basic_types.h>

namespace exanb
{
  
  class GridTriangleIntersections : public OperatorNode
  {
    using Mesh = TriangleMesh<>;
  
    ADD_SLOT( onika::math::AABB            , bounding_box      , INPUT , REQUIRED );
    ADD_SLOT( long                         , resolution        , INPUT , REQUIRED , 128 );
    ADD_SLOT( double                       , distance          , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( Mesh                         , mesh              , INPUT , Mesh{} );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final
    {
      return true;
    }

    inline void execute() override final
    {
      const auto max_dist = *distance;
      
      auto grid_size = bounds_size( *bounding_box );
      const double length = std::max( std::max( grid_size.x , grid_size.y ) , grid_size.z );
      const double cell_size = length / (*resolution);
      const IJK dims = { static_cast<ssize_t>(ceil(grid_size.x/cell_size))
                       , static_cast<ssize_t>(ceil(grid_size.y/cell_size))
                       , static_cast<ssize_t>(ceil(grid_size.z/cell_size)) };
      grid_size = dims * cell_size;
      
      const double max_dist_cells = max_dist / cell_size;
      const auto grid_origin = bounding_box->bmin;
      
      grid_to_triangles->m_grid_dims = dims;
      grid_to_triangles->m_cell_size = cell_size;
      grid_to_triangles->m_origin = grid_origin;
      
      
      ldbg << "grid_origin = " << grid_origin << std::endl;
      ldbg << "grid_size = " << grid_size << std::endl;
      ldbg << "cell_size = " << cell_size << std::endl;
      
      const size_t n_triangles = mesh->triangle_count();
      const size_t n_cells = dims.i * dims.j * dims.k;

      ldbg << "n_cells = " << n_cells << std::endl;
      ldbg << "n_triangles = " << n_triangles << std::endl;

      std::vector< std::vector<size_t> > cell_tri_indices;
      size_t total_tri_indices = 0;

      for(size_t t=0;t<n_triangles;t++)
      {
        auto tri = mesh->triangle(t);
        for(int p=0;p<3;p++) tri[p] = ( tri[p] - grid_origin ) / cell_size;
        const auto tri_bounds = bounds( tri );
        const IJK start = { std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.x-max_dist_cells)))
                          , std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.y-max_dist_cells))) 
                          , std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.z-max_dist_cells))) };
        const IJK end = { std::min(dims.i,static_cast<ssize_t>(ceil(tri_bounds.bmax.x+max_dist_cells))) 
                        , std::min(dims.j,static_cast<ssize_t>(ceil(tri_bounds.bmax.y+max_dist_cells))) 
                        , std::min(dims.k,static_cast<ssize_t>(ceil(tri_bounds.bmax.z+max_dist_cells))) };
        for(int k=start.k;k<end.k;k++)
        for(int j=start.j;j<end.j;j++)
        for(int i=start.i;i<end.i;i++)
        {
          const onika::math::AABB box = { { i-max_dist_cells , j-max_dist_cells , k-max_dist_cells } , { i+1+max_dist_cells , j+1+max_dist_cells , k+1+max_dist_cells } };
          if( intersect( box , tri ) )
          {
            const auto cell_idx = grid_ijk_to_index( dims , {i,j,k} );
            assert( cell_idx >= 0 && cell_idx < n_cells );
            cell_tri_indices[cell_idx].push_back(t);
            ++ total_tri_indices;
          }
        }
      }
      
      ldbg << "triangle-cell insertions = "<<total_tri_indices << std::endl;
      
      auto & cell_triangles = grid_to_triangles->m_cell_triangles;      
      cell_triangles.clear();
      cell_triangles.assign( n_cells + 1 + total_tri_indices , 0 );
      size_t next_tri_idx = n_cells + 1;
      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        cell_triangles[cell_idx] = next_tri_idx;
        for(const auto t : cell_tri_indices[cell_idx])
        {
          cell_triangles[ next_tri_idx ++ ] = t;
        }
      }
      assert( next_tri_idx == ( n_cells + 1 + total_tri_indices ) );
      cell_triangles[n_cells] = next_tri_idx;
    }
  };

   // === register factories ===  
  ONIKA_AUTORUN_INIT(grid_triangle_intersections)
  {
    OperatorNodeFactory::instance()->register_factory("grid_triangle_intersections",make_simple_operator< GridTriangleIntersections >);
  }

}
