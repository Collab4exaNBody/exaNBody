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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/math/basic_types_stream.h>
#include <onika/cuda/input_array_span.h>

#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <exanb/mpi/update_ghost_config.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/update_ghosts_comm_manager.h>
#include <exanb/mpi/grid_update_ghosts.h>
#include <exanb/mpi/update_ghost_functors.h>

#include <mpi.h>
#include <regex>

namespace exanb
{
  template< class GridT, class FieldSetT , bool CreateParticles >
  class UpdateGhostsNodeTmpl;

  template< class GridT, bool CreateParticles, class... fids>
  class UpdateGhostsNodeTmpl< GridT , FieldSet<fids...> , CreateParticles > : public OperatorNode
  {
    using generic_real_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( std::declval<GridT>().field_accessor( field::generic_real{""} ) ) > >;
    using generic_vec3_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( std::declval<GridT>().field_accessor( field::generic_vec3{""} ) ) > >;
    using generic_mat3_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( std::declval<GridT>().field_accessor( field::generic_mat3{""} ) ) > >;
    using UpdateGhostsScratch = UpdateGhostsUtils::UpdateGhostsScratchWithOptionalFields<generic_real_accessor_t,generic_vec3_accessor_t,generic_mat3_accessor_t>;
    using StringVector = std::vector<std::string>;
    using FieldSetT = FieldSet<fids...>;

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( GridT                    , grid              , INPUT_OUTPUT);
    ADD_SLOT( Domain                   , domain            , INPUT );
    ADD_SLOT( GridCellValues           , grid_cell_values  , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( StringVector             , opt_fields        , INPUT , StringVector() , DocString{"List of regular expressions to select optional fields to update"} );

    ADD_SLOT( UpdateGhostConfig        , update_ghost_config , INPUT, UpdateGhostConfig{} );

    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers , PRIVATE );

  public:
    inline void execute() override final
    {
      using onika::cuda::make_input_array_span;
      using onika::cuda::make_const_span;
      constexpr std::integral_constant<size_t,1> embedded_copy_size = {}; // size of embedded array copy held in InputArraySpan to lower array content access latency

      static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");
      static_assert( !CreateParticles || grid_contains_field_set_v<GridT,FieldSetT> , "Creation of ghost particle is not supported for optional fields yet");

      using GridCellValueType = typename GridCellValues::GridCellValueType;
      using CellParticlesUpdateData = typename UpdateGhostsUtils::GhostCellParticlesUpdateData;
      using CellsAccessorT = std::remove_cv_t< std::remove_reference_t< decltype( grid->cells_accessor() ) > >;

      if( ! ghost_comm_scheme.has_value() ) return;
      if( grid->number_of_particles() == 0 ) return;

      // local copy of ghost update config to eventually adapt it to specific constraints
      auto upd_config = *update_ghost_config;

      // automatically assign GPU device id none has been assigned yet
      if( upd_config.gpu_buffer_pack && upd_config.alloc_on_device == nullptr )
      {
        if( global_cuda_ctx()->has_devices() && global_cuda_ctx()->global_gpu_enable() )
        {
          upd_config.alloc_on_device = & ( global_cuda_ctx()->m_devices[0] );
        }
        else
        {
          upd_config.gpu_buffer_pack = false;
          upd_config.alloc_on_device = nullptr;
        }
      }

      auto update_ghost_on_fields = [&]( const auto & update_fields )
      {
        using FieldAccTupleT = std::remove_cv_t< std::remove_reference_t< decltype( update_fields ) > >;
        using PackGhostFunctor = UpdateGhostsUtils::GhostSendPackFunctor<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,FieldAccTupleT>;
        using UnpackGhostFunctor = UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,CreateParticles,FieldAccTupleT>;
        using UpdateGhostsCommManager = UpdateGhostsUtils::UpdateGhostsCommManager<PackGhostFunctor,UnpackGhostFunctor>;

        auto pecfunc = [self=this](auto ... args) { return self->parallel_execution_context(args ...); };
        auto peqfunc = [self=this]() -> onika::parallel::ParallelExecutionQueue& { return self->parallel_execution_queue(); }; 

        if( ghost_comm_buffers->m_comm_resources == nullptr )
        {
          ghost_comm_buffers->m_comm_resources = std::make_shared<UpdateGhostsCommManager>();
        }
        UpdateGhostsCommManager * ghost_scratch = ( UpdateGhostsCommManager * ) ghost_comm_buffers->m_comm_resources.get();

        ldbg << pathname() << " : ";
        print_field_tuple( ldbg , update_fields );
        ldbg<< ", Particle size ="<<onika::soatl::field_id_tuple_size_bytes( update_fields )<< std::endl;

        grid_update_ghosts( ldbg, *mpi, *ghost_comm_scheme, grid.get_pointer(), *domain, grid_cell_values.get_pointer(),
                            * ghost_scratch, pecfunc,peqfunc, update_fields,
                            upd_config, std::integral_constant<bool,CreateParticles>{} );
      };

      // build-up list of field accessors to use for ghost update
      const auto& flist = *opt_fields;
      auto opt_field_upd = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;

      auto & opt_real = ghost_comm_buffers->m_opt_real_fields; opt_real.clear();
      auto & opt_vec3 = ghost_comm_buffers->m_opt_vec3_fields; opt_vec3.clear();
      auto & opt_mat3 = ghost_comm_buffers->m_opt_mat3_fields; opt_mat3.clear();      
      for(const auto & opt_name : grid->optional_scalar_fields()) if(opt_field_upd(opt_name)) { opt_real.push_back( grid->field_accessor(field::mk_generic_real(opt_name)) ); ldbg<<"opt. ghost real "<<opt_name<<std::endl; }
      for(const auto & opt_name : grid->optional_vec3_fields()  ) if(opt_field_upd(opt_name)) { opt_vec3.push_back( grid->field_accessor(field::mk_generic_vec3(opt_name)) ); ldbg<<"opt. ghost vec3 "<<opt_name<<std::endl; }
      for(const auto & opt_name : grid->optional_mat3_fields()  ) if(opt_field_upd(opt_name)) { opt_mat3.push_back( grid->field_accessor(field::mk_generic_mat3(opt_name)) ); ldbg<<"opt. ghost mat3 "<<opt_name<<std::endl; }

      auto update_fields = onika::make_flat_tuple( grid->field_accessor( onika::soatl::FieldId<fids>{} ) ...
                                                 , make_input_array_span(opt_real,embedded_copy_size) 
                                                 , make_input_array_span(opt_vec3,embedded_copy_size) 
                                                 , make_input_array_span(opt_mat3,embedded_copy_size) );
      update_ghost_on_fields( update_fields );
    }

    inline std::string documentation() const override final
    {
      return R"EOF(

Update field value in ghost layers at MPI subdomain and domain boundaries. Usefull when some operators compute a per particle field and that field needs to be updated in the ghosts for visualisation of computation purpose.

Multiple variants are predefined for that operation:
- ghost_update_all: updates all per-particle fields.
- ghost_update_all_no_fv: updates all per-particle fields except forces and velocities.
- ghost_update_r: updates particles positions only.
- ghost_update_opt: updates optional fields created by specific operators. The list of optional fields needs to be passed as a list to that operator ( opt_fields: [ "field1", "field2" ].

Usage example:

dump_data:
  - ghost_update_all
  - average_neighbors_scalar:
      nbh_field: vx
      avg_field: avgvx
      rcut: 8.0 ang
  - ghost_update_opt: { opt_fields: [ "avgvx" ] }
  - write_paraview: { fields: [ "vx", "avgvx" ], write_ghost: true }

)EOF";
    }    
  };

  template< class GridT, class FieldSetT, bool CreateParticles>
  using UpdateGhostsNode = UpdateGhostsNodeTmpl< GridT , FieldSetT , CreateParticles >;
}

