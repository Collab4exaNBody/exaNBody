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
#include <exanb/mesh/particle_mesh_collision.h>
#include <exanb/mesh/edge_math.h>
#include <onika/math/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  struct TestMeshCollisionFunctor
  {
    // in front of an edge, within given maximum distance range
    ONIKA_HOST_DEVICE_FUNC
    inline void operator () (size_t cell_idx, size_t p_idx, size_t tri_idx, const Vec3d& r, const Vec3d& v, double delta_t, double w0, double w1, double w2 ) const
    {
      printf("Cell %d , P %d : pos=(%f,%f,%f) , vel=(%f,%f,%f) : impact triangle %d at t+%f\n",int(cell_idx),int(p_idx),r.x,r.y,r.z,v.x,v.y,v.z,int(tri_idx),delta_t);
    }
  };

  template< class GridT >
  class TriangleMeshCollisionTest : public OperatorNode
  {
    ADD_SLOT( double                       , dt                , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT , OPTIONAL );
    ADD_SLOT( TriangleMesh                 , mesh              , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridT                        , grid              , INPUT_OUTPUT , REQUIRED );

  public:
    inline void execute() override final
    {
      auto rx = grid->field_accessor( field::rx );
      auto ry = grid->field_accessor( field::ry );
      auto rz = grid->field_accessor( field::rz );

      auto vx = grid->field_accessor( field::vx );
      auto vy = grid->field_accessor( field::vy );
      auto vz = grid->field_accessor( field::vz );
      
      const double delta_t = *dt;
      
      if( grid_to_triangles.has_value() )
      {
        GridParticleTriangleCollision<TestMeshCollisionFunctor> func = { read_only_view(*grid_to_triangles) , read_only_view(*mesh) , delta_t };
        compute_cell_particles( *grid , false , func, onika::make_flat_tuple(rx,ry,rz,vx,vy,vz) , parallel_execution_context() );
      }
      else
      {
        TrivialTriangleLocator all_triangles = { { 0 , mesh->triangle_count() } };
        ParticleTriangleCollision<TestMeshCollisionFunctor> func = { all_triangles , read_only_view(*mesh) };
        compute_cell_particles( *grid , false , func, onika::make_flat_tuple(rx,ry,rz,vx,vy,vz) , parallel_execution_context() );
      }

    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(particle_mesh_ineraction)
  {
    OperatorNodeFactory::instance()->register_factory("trimesh_collision",make_grid_variant_operator< TriangleMeshCollisionTest >);
  }

}
