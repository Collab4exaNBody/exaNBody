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

#include <utility>

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <onika/memory/allocator.h>
#include <onika/soatl/field_pointer_tuple.h>

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_cell_projection.h>

namespace exanb
{

  struct DefaultFrictionFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator () ( const Vec3d& r, const Vec3d& pv, const Vec3d& fv, double cx ) const
    {
      const Vec3d relative_velocity = fv - pv;
      const double relative_velocity_norm = norm(relative_velocity);
      return cx * relative_velocity * relative_velocity_norm;
    }
  };


  template<class CellsT, class FrictionFuncT = DefaultFrictionFunctor >
  struct ParticleFluidFrictionFunctor
  {
    const IJK m_dims = {0,0,0};
    const Vec3d m_origin = {0.,0.,0.};
    const double m_cell_size = 0.0;
    const CellsT * m_cells = nullptr;
    
    const double * __restrict__ cell_value_ptr = nullptr;
    const size_t cell_value_stride = 0;
    const unsigned int subdiv = 0;
    const FrictionFuncT friction = {};
    
    template<class... AdditionalArgs>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (
      double rx, double ry, double rz,
      double vx, double vy, double vz,
      double& fx, double& fy, double& fz , const AdditionalArgs& ... args ) const
    {
      using namespace ParticleCellProjectionTools;

      assert( cell_value_stride >= subdiv*subdiv*subdiv * 4 ); // we only accept 4 components vector fields
      const Vec3d r = {rx,ry,rz};
      const Vec3d v = {vx,vy,vz};
      if( cell_value_ptr != nullptr )
      {
        const IJK loc = make_ijk( ( r - m_origin ) / m_cell_size );
        const Vec3d cell_origin = m_origin + ( loc * m_cell_size );
        IJK center_cell_loc = {0,0,0};
        IJK center_subcell_loc = {0,0,0};
        Vec3d rco = r - cell_origin;
        //const double cell_size = grid.cell_size();
        const double subcell_size = m_cell_size / subdiv;
        localize_subcell( rco, m_cell_size, subcell_size, subdiv, center_cell_loc, center_subcell_loc );
        center_cell_loc += loc;
        const ssize_t nbh_cell_i = grid_ijk_to_index( m_dims , center_cell_loc );
        const ssize_t nbh_subcell_i = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , center_subcell_loc );
        //assert( nbh_cell_i>=0 && nbh_cell_i<ssize_t(grid.number_of_cells()) );
        assert( nbh_subcell_i>=0 && nbh_subcell_i<(subdiv*subdiv*subdiv) );

        // access fluid velocity in the grid's sub cell at particle location
        const double cx = cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*4 + 0 ] ;
        const Vec3d flow_velocity = {
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*4 + 1 ] ,
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*4 + 2 ] ,
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*4 + 3 ] };

        // compute friction force
        Vec3d force = friction( r , v , flow_velocity , cx , args ... );
        fx += force.x;
        fy += force.y;
        fz += force.z;
      }

    }
  };

  template<class GridT,class FuncT> struct ComputeCellParticlesTraits< ParticleFluidFrictionFunctor<GridT,FuncT> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT ,
    class FrictionFuncT = DefaultFrictionFunctor ,
    class AdditionalFieldSetT = FieldSet<> ,
    class = AssertGridContainFieldSet< GridT, ConcatFieldSet< FieldSet< field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz > , AdditionalFieldSetT > >
    >
  class FluidFriction : public OperatorNode
  {  
    ADD_SLOT( GridT          , grid             , INPUT_OUTPUT            , DocString{"Grid of cells of particles"} );
    ADD_SLOT( bool           , ghost            , INPUT, false            , DocString{"set to true to enable compute in ghost area"} );
    ADD_SLOT( GridCellValues , grid_cell_values , INPUT, GridCellValues{} , DocString{"Grid cell values, holding cell centered values"} );
    ADD_SLOT( std::string    , field_name       , INPUT , REQUIRED        , DocString{"cell value field holding fluid velocity. must be a 4-component vector (cx, vx,vy,vz)"} );
    ADD_SLOT( FrictionFuncT  , friction_func    , INPUT , FrictionFuncT{} , DocString{"Friction functor"} );

    static constexpr ConcatFieldSet< FieldSet< field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz > , AdditionalFieldSetT > compute_field_set{};
    
  public:
    inline void execute () override final
    {      
      if( grid->number_of_cells() == 0 ) return;

      const auto field = grid_cell_values->field(*field_name);
      const unsigned int subdiv = field.m_subdiv;
      if( field.m_components != subdiv*subdiv*subdiv*4 )
      {
        fatal_error() << "field "<< *field_name <<" must have exactly 4 compnents (cx, vx,vy,vz), but has "<< field.m_components*1.0/(subdiv*subdiv*subdiv) << std::endl;
      }
      const auto field_data = grid_cell_values->field_data(*field_name);

      const auto * cells = grid->cells();
      using CellsT = std::remove_cv_t< std::remove_reference_t< decltype(cells[0]) > >;
      ParticleFluidFrictionFunctor< CellsT , FrictionFuncT> func = { grid->dimension(),grid->origin(),grid->cell_size(),grid->cells() , field_data.m_data_ptr , field_data.m_stride , subdiv , *friction_func };
      
      compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );                  
    }

  };

}
