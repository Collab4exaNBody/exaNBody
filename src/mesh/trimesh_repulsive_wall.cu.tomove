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
#include <exanb/mesh/particle_mesh_proximty_functor.h>
#include <exanb/mesh/edge_math.h>
#include <onika/math/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  template<class CellsAccessorT, bool RepulsiveFeedBack = true>
  struct RepulsiveMeshFunctor
  {
    const double m_rcut;
    const double m_epsilon;
    const int m_exponent;
    CellsAccessorT m_cells;
    Vec3d * m_vertex_force;

    // in front of an edge, within given maximum distance range
    ONIKA_HOST_DEVICE_FUNC
    inline void operator () (const Vec3d& r, const Vec3d& pv, double d, size_t cell_idx, size_t p_idx, ssize_t v0, ssize_t v1, ssize_t v2, double w0, double w1, double w2, bool behind ) const
    {
      const auto N = ( r - pv ) / d;
      const double ratio = 1.0 - m_rcut / d;
      const double ratio_pow_exponent   = pow(ratio,m_exponent);
      double f_contrib =  -m_epsilon * m_exponent * ( m_rcut / (d*d) ) * ratio_pow_exponent / ratio ;
      const Vec3d F = N * f_contrib;

      // particle force update
      ONIKA_CU_ATOMIC_ADD( m_cells[cell_idx][field::fx][p_idx] , F.x );
      ONIKA_CU_ATOMIC_ADD( m_cells[cell_idx][field::fy][p_idx] , F.y );
      ONIKA_CU_ATOMIC_ADD( m_cells[cell_idx][field::fz][p_idx] , F.z );
      
      // mesh vertices force update
      if constexpr ( RepulsiveFeedBack )
      {
        if(w0>0.0 && v0!=-1)
        {
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v0].x , - w0 * F.x );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v0].y , - w0 * F.y );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v0].z , - w0 * F.z );
        }
        if(w1>0.0 && v1!=-1)
        {
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v1].x , - w1 * F.x );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v1].y , - w1 * F.y );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v1].z , - w1 * F.z );
        }
        if(w0>0.0 && v0!=-1)
        {
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v2].x , - w2 * F.x );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v2].y , - w2 * F.y );
          ONIKA_CU_ATOMIC_ADD( m_cellsm_vertex_force[v2].z , - w2 * F.z );
        }
      }
    }

  };

  template< class GridT >
  class TriangleMeshWall : public OperatorNode
  {
    ADD_SLOT( double                       , rcut          , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( double                       , epsilon       , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( long                         , exponent      , INPUT , REQUIRED , 1.0 );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT , OPTIONAL );
    
    ADD_SLOT( TriangleMesh                 , mesh              , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( VertexAray                   , vertex_force      , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( GridT                        , grid              , INPUT_OUTPUT , REQUIRED );

  public:
    inline void execute() override final
    {
      vertex_force->assign( mesh->vertex_count() , Vec3d{0,0,0} );

      auto rx = grid->field_accessor( field::rx );
      auto ry = grid->field_accessor( field::ry );
      auto rz = grid->field_accessor( field::rz );
      auto cells_accessor = grid->cells_accessor();

      auto execute_repulsive_wall = [&] ( auto enable_force_feedback , Vec3d * mesh_force_feedback )
      {
        using FuncT = RepulsiveMeshFunctor<decltype(cells_accessor),enable_force_feedback.value>;
        FuncT mesh_wall = { *rcut , *epsilon, static_cast<int>(*exponent), cells_accessor , mesh_force_feedback };
        ParticleMeshProximityFunctor<FuncT,true,true> func = { read_only_view(*grid_to_triangles) , read_only_view(*mesh) , *rcut , mesh_wall };
        compute_cell_particles( *grid , false , func, onika::make_flat_tuple(rx,ry,rz) , parallel_execution_context() );
      };
      
      if( vertex_force.has_value() )
        execute_repulsive_wall( std::true_type{}  , vertex_force->data() );
      else 
        execute_repulsive_wall( std::false_type{} , nullptr );
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(particle_mesh_ineraction)
  {
    OperatorNodeFactory::instance()->register_factory("trimesh_wall",make_grid_variant_operator< TriangleMeshWall >);
  }

}
