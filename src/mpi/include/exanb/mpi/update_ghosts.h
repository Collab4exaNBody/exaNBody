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
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>
#include <regex>

#include <mpi.h>
#include <exanb/mpi/grid_update_ghosts.h>
#include <onika/mpi/data_types.h>

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>

namespace exanb
{
  template< class GridT, class FieldSetT , bool CreateParticles >
  class UpdateGhostsNodeTmpl;

  template< class GridT, bool CreateParticles, class... fids>
  class UpdateGhostsNodeTmpl< GridT , FieldSet<fids...> , CreateParticles > : public OperatorNode
  {
    using UpdateGhostsScratch = UpdateGhostsUtils::UpdateGhostsScratch;
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
    ADD_SLOT( long                     , mpi_tag           , INPUT , 0 );
    ADD_SLOT( StringVector             , opt_fields        , INPUT , StringVector() , DocString{"List of regular expressions to select optional fields to update"} );

    ADD_SLOT( bool                     , gpu_buffer_pack   , INPUT , false );
    ADD_SLOT( bool                     , async_buffer_pack , INPUT , false );
    ADD_SLOT( bool                     , staging_buffer    , INPUT , false );
    ADD_SLOT( bool                     , serialize_pack_send , INPUT , false );
    ADD_SLOT( bool                     , wait_all          , INPUT , false );

    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers, PRIVATE );

  public:
    inline void execute() override final
    {
      if( ! ghost_comm_scheme.has_value() ) return;
      if( grid->number_of_particles() == 0 ) return;

      using onika::cuda::make_const_span;
      
      const auto& flist = *opt_fields;
      auto opt_field_upd = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;

      using generic_real_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( grid->field_accessor( field::generic_real{""} ) ) > >;
      using generic_vec3_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( grid->field_accessor( field::generic_vec3{""} ) ) > >;
      using generic_mat3_accessor_t =  std::remove_cv_t< std::remove_reference_t< decltype( grid->field_accessor( field::generic_mat3{""} ) ) > >;
      std::vector< generic_real_accessor_t > opt_real;
      std::vector< generic_vec3_accessor_t > opt_vec3;
      std::vector< generic_mat3_accessor_t > opt_mat3;
      for(const auto & opt_name : grid->optional_scalar_fields()) if(opt_field_upd(opt_name)) { opt_real.push_back( grid->field_accessor(field::mk_generic_real(opt_name)) ); ldbg<<"opt. ghost real "<<opt_name<<std::endl; }
      for(const auto & opt_name : grid->optional_vec3_fields()  ) if(opt_field_upd(opt_name)) { opt_vec3.push_back( grid->field_accessor(field::mk_generic_vec3(opt_name)) ); ldbg<<"opt. ghost vec3 "<<opt_name<<std::endl; }
      for(const auto & opt_name : grid->optional_mat3_fields()  ) if(opt_field_upd(opt_name)) { opt_mat3.push_back( grid->field_accessor(field::mk_generic_mat3(opt_name)) ); ldbg<<"opt. ghost mat3 "<<opt_name<<std::endl; }

      auto pecfunc = [self=this](auto ... args) { return self->parallel_execution_context(args ...); };
      auto peqfunc = [self=this]() -> onika::parallel::ParallelExecutionQueue& { return self->parallel_execution_queue(); }; 
      static_assert( !CreateParticles || grid_contains_field_set_v<GridT,FieldSetT> , "Creation of ghost particle is not supported for optional fields yet");
      
      auto update_fields = onika::make_flat_tuple( grid->field_accessor( onika::soatl::FieldId<fids>{} ) ... , make_const_span(opt_real) , make_const_span(opt_vec3) , make_const_span(opt_mat3) );

      grid_update_ghosts( ldbg, *mpi, *ghost_comm_scheme, grid.get_pointer(), *domain, grid_cell_values.get_pointer(),
                          *ghost_comm_buffers, pecfunc,peqfunc, update_fields,
                          *mpi_tag, *gpu_buffer_pack, *async_buffer_pack, *staging_buffer,
                          *serialize_pack_send, *wait_all, std::integral_constant<bool,CreateParticles>{} );
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

