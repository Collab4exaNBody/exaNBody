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

#include <onika/math/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/mpi/update_from_ghost_utils.h>
#include <onika/mpi/data_types.h>
#include <exanb/mpi/update_ghost_config.h>

#include <exanb/grid_cell_particles/cell_particle_update_functor.h>

#include <onika/soatl/field_tuple.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/soatl/field_id_tuple_utils.h>

#include <vector>
#include <string>
#include <algorithm>

#include <mpi.h>

namespace exanb
{

/*
  This function updates some or all of the internal particle fields using values of neighbor subdomain's ghost particles
  This done following a previously computed communication scheme in reverse order. UpdateFuncT is the type of the functor
  responible to merge existing values with incoming values, default is simple addition.
  Communication consists of asynchronous MPI sends and receives, as well as packing and unpacking of data messages.
  Executes roughly as follows :
  1. parallel pack messages to be sent (optionally using GPU)
  2. asynchronous sends messages
  3. launch asynchronous receives
  4. while packets to be received, wait for some message tio be received
      4.a asynchronous, parallel unpack received message (optionally used the GPU)
      4.b free send messages resources as acknowledgements for sent messages are received
  options :
  staging_buffer option requires to perform a CPU copy to a CPU allocated buffer of messages to be sent and received messages to be unpacked, in case of a non GPU-Aware MPi implementation
  serialize_pack_sends requires to wait until all send packets are filled before starting to send the first one
  gpu_packing allows pack/unpack operations to execute on the GPU
  */
  template<class LDBGT, class GridT, class UpdateGhostsScratchT, class PECFuncT, class PEQFuncT, class FieldAccTupleT, bool CreateParticles=false>
  static inline void grid_update_from_ghosts(
    LDBGT& ldbg,
    MPI_Comm comm,
    GhostCommunicationScheme& comm_scheme,
    GridT* gridp,
    const Domain& domain,
    GridCellValues* grid_cell_values,
    UpdateGhostsScratchT& ghost_comm_buffers,
    const PECFuncT& parallel_execution_context,
    const PEQFuncT& parallel_execution_queue,
    const FieldAccTupleT& update_fields,
    const UpdateGhostConfig& config,
    std::integral_constant<bool,CreateParticles> create_cell_particles = {}
    )
  {
    auto [alloc_on_device,comm_tag,gpu_buffer_pack,async_buffer_pack,staging_buffer,serialize_pack_send,wait_all] = config;

    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using CellParticlesUpdateData = typename UpdateGhostsUtils::GhostCellParticlesUpdateData;
    
    static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    using CellsAccessorT = std::remove_cv_t< std::remove_reference_t< decltype( gridp->cells_accessor() ) > >;
    using PackGhostFunctor = typename UpdateGhostsScratchT::PackGhostFunctor;
    using UnpackGhostFunctor = typename UpdateGhostsScratchT::UnpackGhostFunctor;
    using ParForOpts = onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;

    if( create_cell_particles && gridp==nullptr )
    {
      fatal_error() << "request for ghost particle creation while null grid passed in"<< std::endl;
    }

    const size_t sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );

    int nprocs = 1;
    int rank = 0;
    MPI_Comm_size(comm,&nprocs);
    MPI_Comm_rank(comm,&rank);

    const size_t ghost_layers = (gridp!=nullptr) ? gridp->ghost_layers() : ( (grid_cell_values!=nullptr) ? grid_cell_values->ghost_layers() : 0 );
    const IJK grid_dims = (gridp!=nullptr) ? gridp->dimension() : ( (grid_cell_values!=nullptr) ? grid_cell_values->grid_dims() : IJK{0,0,0} );
    const IJK grid_domain_offset = (gridp!=nullptr) ? gridp->offset() : ( (grid_cell_values!=nullptr) ? grid_cell_values->grid_offset() : IJK{0,0,0} );
    double cell_size = domain.cell_size();
    const Vec3d grid_start_position = domain.origin() + ( grid_domain_offset * cell_size );
    const size_t n_cells = (gridp!=nullptr) ? gridp->number_of_cells() : ( (grid_cell_values!=nullptr) ? grid_cell_values->number_of_cells() : 0 );

    if( gridp!=nullptr )
    {
      assert( n_cells == gridp->number_of_cells() );
      assert( ghost_layers == gridp->ghost_layers() );
      assert( grid_dims == gridp->dimension() );
      assert( grid_domain_offset == gridp->offset() );
      assert( grid_start_position == gridp->cell_position({0,0,0}) );
    }
    if( grid_cell_values!=nullptr )
    {
      assert( n_cells == grid_cell_values->number_of_cells() );
      assert( ghost_layers == grid_cell_values->ghost_layers() );
      assert( grid_dims == grid_cell_values->grid_dims() );
      assert( grid_domain_offset == grid_cell_values->grid_offset() );
    }
    ldbg<<"grid_update_ghosts : n_cells="<<n_cells<<", ghost_layers="<<ghost_layers<<", grid_dims="<<grid_dims
        <<", grid_domain_offset="<<grid_domain_offset<<", grid_start_position="<<grid_start_position
        <<", cell_size="<<cell_size<< ", sizeof_ParticleTuple="<<sizeof_ParticleTuple<<std::endl;

    auto * const cells = (gridp!=nullptr) ? gridp->cells() : nullptr;
    const GhostBoundaryModifier ghost_boundary = { domain.origin() , domain.extent() };

    // per cell scalar values, if any
    GridCellValueType* cell_scalars = nullptr;
    unsigned int cell_scalar_components = 0;
    if( grid_cell_values != nullptr  )
    {
      cell_scalar_components = grid_cell_values->components();
      if( cell_scalar_components > 0 )
      {
        assert( grid_cell_values->data().size() == n_cells * cell_scalar_components );
        cell_scalars = grid_cell_values->data().data();
      }
    }
    if( cell_scalars )
    {
      ldbg << "update ghost cell values with "<< cell_scalar_components << " components"<<std::endl;
    }
    else
    {
      cell_scalar_components = 0;
    }

    CellsAccessorT cells_accessor = { nullptr , nullptr };
    if( gridp != nullptr ) cells_accessor = gridp->cells_accessor();

    // ***************** send/receive buffers resize ******************
    ghost_comm_buffers.update_from_comm_scheme( rank, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, alloc_on_device, staging_buffer, async_buffer_pack );
    ghost_comm_buffers.reactivate_requests();

    // ***************** resize cells if needed ******************
    ghost_comm_buffers.resize_received_cells( cells, gridp->cell_allocator(), create_cell_particles );

    // ***************** send bufer packing start ******************
    uint8_t* send_buf_ptr = ghost_comm_buffers.mpi_recv_buffer();
    int active_send_packs = ghost_comm_buffers.start_pack_functors( parallel_execution_queue, parallel_execution_context, rank, gpu_buffer_pack );

    // number of flying messages
    int active_sends = 0;
    int active_recvs = 0;
    int request_count = 0;

    // ***************** async receive start ******************
    request_count = active_recvs = ghost_comm_buffers.start_mpi_receives(comm,comm_tag,rank);

    // ***************** initiate buffer sends ******************
    if( serialize_pack_send ) parallel_execution_queue().wait();
    while( active_sends < active_send_packs )
    {
      active_sends += ghost_comm_buffers.start_ready_mpi_sends(parallel_execution_queue, comm, comm_tag, rank, serialize_pack_send);
    }
    request_count += active_sends;

    assert( ghost_comm_buffers.number_of_requests() == request_count );
    assert( ghost_comm_buffers.number_of_requests() == ( active_sends + active_recvs ) );
    ldbg << "UpdateGhosts : total active requests = "<<ghost_comm_buffers.number_of_requests()<<std::endl;

    // manage loopback communication : decode packet directly from sendbuffer without actually receiving it
    size_t ghost_cells_self=0;
    if( ghost_comm_buffers.send_info(rank).buffer_size > 0 || ghost_comm_buffers.recv_info(rank).buffer_size > 0 )
    {
      assert( ghost_comm_buffers.send_info(rank).buffer_size == ghost_comm_buffers.recv_info(rank).buffer_size );
      ldbg << "UpdateGhosts: loopback buffer size="<<ghost_comm_buffers.send_info(rank).buffer_size<<std::endl;
      if( ! serialize_pack_send ) { parallel_execution_queue().wait( ghost_comm_buffers.pack_functor_lane[rank] ); }
      ghost_cells_self = ghost_comm_buffers.process_received_buffer(parallel_execution_queue, parallel_execution_context,rank,gpu_buffer_pack );
    }

    const size_t ghost_cells_recv = ghost_comm_buffers.wait_mpi_messages(parallel_execution_queue, parallel_execution_context,rank,wait_all,gpu_buffer_pack);

    for(int p=0;p<nprocs;p++)
    {
      parallel_execution_queue().wait( ghost_comm_buffers.unpack_functor_lane[p] );
    }

    if( create_cell_particles )
    {
      gridp->rebuild_particle_offsets();
    }

    ldbg << "--- end update_ghosts : received "<<ghost_cells_recv<<" cells and loopbacked "<<ghost_cells_self<<" cells"<< std::endl;
  }

}

