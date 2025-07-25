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
#include <exanb/mesh/particle_mesh_proximty.h>
#include <exanb/mesh/edge_math.h>
#include <onika/math/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  struct TestMeshProximityFunctor
  {
    // in front of an edge, within given maximum distance range
    ONIKA_HOST_DEVICE_FUNC
    inline void operator () (const Vec3d& r, const Vec3d& pv, double d, size_t cell_idx, size_t p_idx, ssize_t v0, ssize_t v1, ssize_t v2, double w0, double w1, double w2, bool behind ) const
    {
      if( v1!=-1 && v2!=-1 )
      {
        printf("Cell %d , P %d <-> Triangle{%d;%d;%d} : d=%f , behind=%d\n",int(cell_idx),int(p_idx),int(v0),int(v1),int(v2),d,int(behind));
      }
      else if( v1!=-1 )
      {
        printf("Cell %d , P %d <-> Edge{%d;%d} : d=%f\n",int(cell_idx),int(p_idx),int(v0),int(v1),d);
      }
      else
      {
        printf("Cell %d , P %d <-> Vertex{%d} : d=%f\n",int(cell_idx),int(p_idx),int(v0),d);
      }
    }
  };

  template< class GridT >
  class TriangleMeshProximityTest : public OperatorNode
  {
    ADD_SLOT( double                       , max_dist          , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT , OPTIONAL );
    ADD_SLOT( TriangleMesh                 , mesh              , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridT                        , grid              , INPUT_OUTPUT , REQUIRED );

  public:
    inline void execute() override final
    {
      auto rx = grid->field_accessor( field::rx );
      auto ry = grid->field_accessor( field::ry );
      auto rz = grid->field_accessor( field::rz );

      bool fallback_to_trivial_locator = true;
      if( grid_to_triangles.has_value() )
      {
        if( ! grid_to_triangles->exceeds_maximum_distance(*max_dist) )
        {
          fallback_to_trivial_locator = false;
          GridParticleTriangleProjProximity<TestMeshProximityFunctor,true,true> func = { read_only_view(*grid_to_triangles) , read_only_view(*mesh) , *max_dist };
          compute_cell_particles( *grid , false , func, onika::make_flat_tuple(rx,ry,rz) , parallel_execution_context() );
        }
      }
      if( fallback_to_trivial_locator )
      {
        TrivialTriangleLocator all_triangles = { { 0 , mesh->triangle_count() } };
        ParticleTriangleProjProximity<TestMeshProximityFunctor,true,true> func = { all_triangles , read_only_view(*mesh) , *max_dist };
        compute_cell_particles( *grid , false , func, onika::make_flat_tuple(rx,ry,rz) , parallel_execution_context() );
      }

    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(trimesh_proximity)
  {
    OperatorNodeFactory::instance()->register_factory("trimesh_proximity",make_grid_variant_operator< TriangleMeshProximityTest >);
  }

}
