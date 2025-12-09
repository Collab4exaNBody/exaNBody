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
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <regex>

#include <mpi.h>
#include <exanb/mpi/update_from_ghost_utils.h>
#include <exanb/mpi/grid_update_from_ghosts.h>
#include <onika/mpi/data_types.h>
#include <exanb/grid_cell_particles/cell_particle_update_functor.h>

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>

namespace exanb
{

  template< class GridT, class FieldSetT, class UpdateFuncT = UpdateValueAdd >
  class UpdateFromGhostsTmpl;

  template< class GridT, class UpdateFuncT, class... fids >
  class UpdateFromGhostsTmpl< GridT , FieldSet<fids...> , UpdateFuncT > : public OperatorNode
  {  
    using FieldSetT = FieldSet<fids...>;
    using CellParticles = typename GridT::CellParticles;
    using UpdateGhostsScratch = UpdateGhostsUtils::UpdateGhostsScratch;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using UpdateValueFunctor = UpdateFuncT;
    using StringVector = std::vector<std::string>;
        
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi                , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme  , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( GridT                    , grid               , INPUT_OUTPUT);
    ADD_SLOT( Domain                   , domain             , INPUT );
    ADD_SLOT( GridCellValues           , grid_cell_values   , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( StringVector             , opt_fields         , INPUT , StringVector() , DocString{"List of regular expressions to select optional fields to update"} );

    ADD_SLOT( UpdateGhostConfig        , update_ghost_config, INPUT, UpdateGhostConfig{} );

    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers , PRIVATE );

  public:
    // implementing generate_tasks instead of execute allows to launch asynchronous block_parallel_for, even with OpenMP backend
    inline void execute() override final
    {
      using GridCellValueType = typename GridCellValues::GridCellValueType;
      using UpdateValueFunctor = UpdateFuncT;
      using CellParticlesUpdateData = typename UpdateGhostsUtils::GhostCellParticlesUpdateData;
      static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
      static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");
      using CellsAccessorT = std::remove_cv_t< std::remove_reference_t< decltype( grid->cells_accessor() ) > >;

      if( ! ghost_comm_scheme.has_value() ) return;
      if( grid->number_of_particles() == 0 ) return;
      if( ! grid->has_allocated_fields( FieldSetT{} ) )
      {
        fatal_error() << "Attempt to use UpdateFromGhosts on uninitialized fields" << std::endl;
      }

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
      auto update_fields = onika::make_flat_tuple( grid->field_accessor( onika::soatl::FieldId<fids>{} ) ... , make_const_span(opt_real) , make_const_span(opt_vec3) , make_const_span(opt_mat3) );

      using FieldAccTupleT = std::remove_cv_t< std::remove_reference_t< decltype( update_fields ) > >;
      using PackGhostFunctor = UpdateFromGhostsUtils::GhostReceivePackToSendBuffer<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,FieldAccTupleT>;
      using UnpackGhostFunctor = UpdateFromGhostsUtils::GhostSendUnpackFromReceiveBuffer<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,UpdateFuncT,FieldAccTupleT>;
      using UpdateGhostsCommManager = UpdateGhostsUtils::UpdateGhostsCommManager<PackGhostFunctor,UnpackGhostFunctor>;
      if( ghost_comm_buffers->m_comm_resources == nullptr )
      {
        ghost_comm_buffers->m_comm_resources = std::make_shared<UpdateGhostsCommManager>();
      }
      UpdateGhostsCommManager * ghost_scratch = ( UpdateGhostsCommManager * ) ghost_comm_buffers->m_comm_resources.get();

      ldbg << pathname() << " : ";
      print_field_tuple( ldbg , update_fields );
      ldbg<< ", Particle size ="<<onika::soatl::field_id_tuple_size_bytes( update_fields )<< std::endl;

      grid_update_from_ghosts( ldbg, *mpi, *ghost_comm_scheme, grid.get_pointer(), *domain, grid_cell_values.get_pointer(),
                        *ghost_scratch, pecfunc,peqfunc, update_fields,*update_ghost_config, UpdateFuncT{} );
    }

  };

  template< class GridT , class FieldSetT , class UpdateFuncT = UpdateValueAdd >
  using UpdateFromGhosts = UpdateFromGhostsTmpl< GridT , FieldSetT , UpdateFuncT >;

}

