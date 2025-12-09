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
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/domain.h>
#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <onika/mpi/data_types.h>
#include <exanb/mpi/update_ghost_config.h>

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/soatl/field_id_tuple_utils.h>

namespace exanb
{

  /*
  This function updates some or all of the particle fields in ghost particles (i.e particles in surrounding ghost cells)
  This done following a previously computed communication scheme that tells wich cell/particle goes to which neighbor MPI process.
  Communication consists of asynchronous MPI sends and receives, as well as packing and unpacking of data messages.
  Executes roughly as follows :
  1. parallel pack messages to be sent (optionally using GPU)
  2. asynchronous sends messages
  3. launch asynchronous receives
  4. while packets to be received, wait for some message tio be received
      4.a asynchronous, parallel unpack received message (optionally used the GPU)
      4.a.2 resize cell if ghost particles are created for the first time
      4.b free send messages resources as acknowledgements for sent messages are received
  options :
  staging_buffer option requires to perform a CPU copy to a CPU allocated buffer of messages to be sent and received messages to be unpacked, in case of a non GPU-Aware MPi implementation
  serialize_pack_sends requires to wait until all send packets are filled before starting to send the first one
  gpu_packing allows pack/unpack operations to execute on the GPU
  */
  template<class LDBGT, class GridT, class UpdateGhostsScratchT, class PECFuncT, class PEQFuncT, bool CreateParticles , class FieldAccTupleT>
  static inline void grid_update_ghosts(
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
    std::integral_constant<bool,CreateParticles> )
  {
    auto [_alloc_on_device,comm_tag,_gpu_buffer_pack,async_buffer_pack,staging_buffer,serialize_pack_send,wait_all] = config;
    auto alloc_on_device = _alloc_on_device;
    auto gpu_buffer_pack = _gpu_buffer_pack;
    if( CreateParticles )
    {
      alloc_on_device = nullptr;
      gpu_buffer_pack = false;
    }

    using CellParticles = typename GridT::CellParticles;
    using ParticleFullTuple = typename CellParticles::TupleValueType;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using CellParticlesUpdateData = typename UpdateGhostsUtils::GhostCellParticlesUpdateData;
    
    static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    using CellsAccessorT = std::remove_cv_t< std::remove_reference_t< decltype( gridp->cells_accessor() ) > >;
    using PackGhostFunctor = typename UpdateGhostsScratchT::PackGhostFunctor;
    using UnpackGhostFunctor = typename UpdateGhostsScratchT::UnpackGhostFunctor;
    using ParForOpts = onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;

    if( CreateParticles && gridp==nullptr )
    {
      fatal_error() << "request for ghost particle creation while null grid passed in"<< std::endl;
    }

    const size_t sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );

    //int comm_tag = *mpi_tag;
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

    CellParticles * const cells = (gridp!=nullptr) ? gridp->cells() : nullptr;
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
    ghost_comm_buffers.update_from_comm_scheme( rank, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, alloc_on_device, staging_buffer );
    ghost_comm_buffers.reactivate_requests();

    // ***************** send bufer packing start ******************
    //std::vector<PackGhostFunctor> m_pack_functors( nprocs , PackGhostFunctor{} );
    uint8_t* send_buf_ptr = ghost_comm_buffers.mpi_send_buffer(); // either packet encoding buffer or staging send buffer depending on 'staging_buffer'
    int active_send_packs = 0;
    unsigned int parallel_concurrent_lane = 0;
    for(int p=0;p<nprocs;p++)
    {
      auto & send_info = ghost_comm_buffers.send_info(p);
      if( send_info.buffer_size > 0 )
      {
        assert( send_info.buffer_offset + send_info.buffer_size <= ghost_comm_buffers.sendbuf_total_size() );
        if( p != rank ) { ++ active_send_packs; }

        const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
        // ghost_comm_buffers.pack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
        assert( ghost_comm_buffers.pack_functors[p].ready_for_execution() );

        ParForOpts par_for_opts = {}; par_for_opts.enable_gpu = (!CreateParticles) && gpu_buffer_pack ;
        auto parallel_op = block_parallel_for( cells_to_send, ghost_comm_buffers.pack_functors[p], parallel_execution_context("send_pack") , par_for_opts );

        if( async_buffer_pack )
        {
          send_info.async_lane = parallel_concurrent_lane++;
          parallel_execution_queue() << onika::parallel::set_lane( send_info.async_lane ) << std::move(parallel_op) << onika::parallel::flush;
        }
        else
        {
          send_info.async_lane = onika::parallel::UNDEFINED_EXECUTION_LANE;
        }
      }
    }

    // number of flying messages
    int active_sends = 0;
    int active_recvs = 0;
    int request_count = 0;

    // ***************** async receive start ******************
    uint8_t* recv_buf_ptr = ghost_comm_buffers.mpi_recv_buffer(); // either packet decoding buffer or staging recv buffer depending on 'staging_buffer'
    for(int p=0;p<nprocs;p++)
    {
      auto & recv_info = ghost_comm_buffers.recv_info(p);
      if( p!=rank && recv_info.buffer_size > 0 )
      {
        assert( recv_info.request_idx != -1 );
        assert( ghost_comm_buffers.request_index_is_recv(recv_info.request_idx) );
        assert( recv_info.buffer_offset + recv_info.buffer_size <= ghost_comm_buffers.recvbuf_total_size() );
        assert( ghost_comm_buffers.partner_rank_from_request_index(recv_info.request_idx) == p );
        MPI_Irecv( (char*) recv_buf_ptr + recv_info.buffer_offset, recv_info.buffer_size, MPI_CHAR, p, comm_tag, comm, ghost_comm_buffers.request_ptr(recv_info.request_idx) );
        ++ active_recvs;
        ++ request_count;
      }
    }

    // ***************** initiate buffer sends ******************
    if( serialize_pack_send )
    {
      for(int p=0;p<nprocs;p++) { parallel_execution_queue().wait( ghost_comm_buffers.send_info(p).async_lane ); }
    }
    //std::vector<bool> message_sent( nprocs , false );
    while( active_sends < active_send_packs )
    {
      for(int p=0;p<nprocs;p++)
      {
        auto & send_info = ghost_comm_buffers.send_info(p);
        if( p!=rank && !ghost_comm_buffers.message_sent[p] && send_info.buffer_size > 0 )
        {
          bool ready = true;
          if( ! serialize_pack_send ) { ready = parallel_execution_queue().query_status( send_info.async_lane ); }
          if( ready )
          {
            MPI_Isend( (char*) send_buf_ptr + send_info.buffer_offset , send_info.buffer_size, MPI_CHAR, p, comm_tag, comm, ghost_comm_buffers.request_ptr(send_info.request_idx) );
            assert( ghost_comm_buffers.partner_rank_from_request_index(send_info.request_idx) == p );
            assert( ghost_comm_buffers.request_index_is_send(send_info.request_idx) );
            ++ active_sends;
            ++ request_count;
            ghost_comm_buffers.message_sent[p] = true;
          }
        }
      }
    }

    assert( ghost_comm_buffers.number_of_requests() == request_count );
    assert( ghost_comm_buffers.number_of_requests() == ( active_sends + active_recvs ) );
    ldbg << "UpdateGhosts : total active requests = "<<ghost_comm_buffers.number_of_requests()<<std::endl;

    //std::vector<UnpackGhostFunctor> unpack_functors( nprocs , UnpackGhostFunctor{} );
    size_t ghost_cells_recv=0 , ghost_cells_self=0;

    // *** packet decoding process lambda ***
    auto process_receive_buffer = [&](int p)
    {
      const size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();        
      auto & recv_info = ghost_comm_buffers.recv_info(p);

      // re-allocate cells before they receive particles
      if( CreateParticles )
      {
        for( auto cell_input_it : comm_scheme.m_partner[p].m_receives )
        {
          const auto cell_input = ghost_cell_receive_info(cell_input_it);
          const size_t n_particles = cell_input.m_n_particles;
          const size_t cell_i = cell_input.m_cell_i;
          assert( gridp->is_ghost_cell(cell_i) );
          assert( /*cell_i>=0 &&*/ cell_i<n_cells );
          assert( cells[cell_i].empty() );
          cells[cell_i].resize( n_particles , gridp->cell_allocator() );
        }
      }
      
      if( p != rank ) ghost_cells_recv += cells_to_receive;
      else ghost_cells_self += cells_to_receive;

      // ghost_comm_buffers.unpack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
      assert( ghost_comm_buffers.unpack_functors[p].ready_for_execution() );
      
      ParForOpts par_for_opts = {}; par_for_opts.enable_gpu = (!CreateParticles) && gpu_buffer_pack ;
      auto parallel_op = block_parallel_for( cells_to_receive, ghost_comm_buffers.unpack_functors[p], parallel_execution_context("recv_unpack") , par_for_opts );

      if( async_buffer_pack )
      {
        recv_info.async_lane = parallel_concurrent_lane ++;
        parallel_execution_queue() << onika::parallel::set_lane( recv_info.async_lane ) << std::move(parallel_op) << onika::parallel::flush;
      }
      else
      {
        recv_info.async_lane = onika::parallel::UNDEFINED_EXECUTION_LANE;
      }
    };
    // *** end of packet decoding lamda ***

    // manage loopback communication : decode packet directly from sendbuffer without actually receiving it
    if( ghost_comm_buffers.send_info(rank).buffer_size > 0 )
    {
      if( ghost_comm_buffers.send_info(rank).buffer_size != ghost_comm_buffers.recv_info(rank).buffer_size )
      {
        fatal_error() << "UpdateFromGhosts: inconsistent loopback communictation : send="<<ghost_comm_buffers.send_info(rank).buffer_size<<" receive="<<ghost_comm_buffers.recv_info(rank).buffer_size<<std::endl;
      }
      ldbg << "UpdateGhosts: loopback buffer size="<<ghost_comm_buffers.send_info(rank).buffer_size<<std::endl;
      if( ! serialize_pack_send ) { parallel_execution_queue().wait( ghost_comm_buffers.send_info(rank).async_lane ); }
      process_receive_buffer(rank);
    }

    if( wait_all )
    {
      ldbg << "UpdateFromGhosts: MPI_Waitall, active_requests / total_requests = "<<ghost_comm_buffers.number_of_active_requests()
           <<" / "<<ghost_comm_buffers.number_of_requests() <<std::endl;
      MPI_Waitall( ghost_comm_buffers.number_of_active_requests() , ghost_comm_buffers.requests_data() , MPI_STATUS_IGNORE );
      ldbg << "UpdateFromGhosts: MPI_Waitall done"<<std::endl;
    }
    
    while( active_sends>0 || active_recvs>0 )
    {
      int reqidx = MPI_UNDEFINED;

      if( request_count != ( active_sends + active_recvs ) )
      {
        fatal_error() << "Inconsistent active_requests ("<<request_count<<") != ( "<<active_sends<<" + "<<active_recvs<<" )" <<std::endl;
      }
      ldbg << "UpdateGhosts: "<< active_sends << " SENDS, "<<active_recvs<<" RECVS, (total "<<request_count<<") :" ;
      for(int i=0;i<request_count;i++)
      {
        ldbg << " REQ"<< i << "="<< ( ghost_comm_buffers.request_index_is_recv(i) ? "RECV-P" : "SEND-P" ) << ghost_comm_buffers.partner_rank_from_request_index(i);
      }
      ldbg << std::endl;

      if( wait_all )
      {
        reqidx = request_count-1; // process communications in reverse order
      }
      else
      {
        assert( request_count == ghost_comm_buffers.number_of_active_requests() );
        if( request_count == 1 )
        {
          MPI_Wait( ghost_comm_buffers.requests_data() , MPI_STATUS_IGNORE );
          reqidx = 0;
        }
        else
        {
          MPI_Waitany( request_count , ghost_comm_buffers.requests_data() , &reqidx , MPI_STATUS_IGNORE );
        }
      }

      if( reqidx != MPI_UNDEFINED )
      {
        if( reqidx<0 || reqidx >= request_count )
        {
          fatal_error() << "bad request index "<<reqidx<<std::endl;
        }
        const int p = ghost_comm_buffers.partner_rank_from_request_index(reqidx); 
        assert( p >= 0 && p < nprocs );
        if( ghost_comm_buffers.request_index_is_recv(reqidx) ) // it's a receive
        {
          process_receive_buffer(p);
          -- active_recvs;
        }
        else // it's a send
        {
          -- active_sends;
        }

        ghost_comm_buffers.deactivate_request( reqidx );
        -- request_count;
        assert( request_count == ghost_comm_buffers.number_of_active_requests() );
      }
      else
      {
        ldbg << "Warning: undefined request index returned by MPI_Waitany"<<std::endl;
      }
    }      

    for(int p=0;p<nprocs;p++)
    {
      parallel_execution_queue().wait( ghost_comm_buffers.recv_info(p).async_lane );
    }

    if( CreateParticles )
    {
      gridp->rebuild_particle_offsets();
    }

#if 0
    {
      static constexpr const char* memTypeStr[] = { "Unregistered" , "Host" , "Device" , "Managed" };
      auto * sendbuf_ptr = ghost_comm_buffers.sendbuf_ptr(0);
      auto * recvbuf_ptr = ghost_comm_buffers.recvbuf_ptr(0);
      cudaPointerAttributes info;
      cudaPointerGetAttributes( &info , sendbuf_ptr );
      lout << "grid_update_ghosts: sendbuf: device="<<info.device<<", devPtr="<<info.devicePointer<<", hostPtr="<<info.hostPointer<<", type="<<memTypeStr[info.type]<<std::flush;
      cudaPointerGetAttributes( &info , recvbuf_ptr );
      lout << ", recvbuf: device="<<info.device<<", devPtr="<<info.devicePointer<<", hostPtr="<<info.hostPointer<<", type="<<memTypeStr[info.type]<<std::endl;
    }
#endif

    ldbg << "--- end update_ghosts : received "<<ghost_cells_recv<<" cells and loopbacked "<<ghost_cells_self<<" cells"<< std::endl;  
  }

  // version with a Grid reference instead of a pointer, for backward compatibility
  template<class LDBGT, class GridT, class UpdateGhostsScratchT, class PECFuncT, class PEQFuncT, bool CreateParticles , class FieldAccTupleT>
  static inline void grid_update_ghosts(
    LDBGT& ldbg,
    MPI_Comm comm,
    GhostCommunicationScheme& comm_scheme,
    GridT& grid,
    const Domain& domain,
    GridCellValues* grid_cell_values,
    UpdateGhostsScratchT& ghost_comm_buffers,
    const PECFuncT& parallel_execution_context,
    const PEQFuncT& parallel_execution_queue,
    const FieldAccTupleT& update_fields,
    long comm_tag ,
    bool gpu_buffer_pack ,
    bool async_buffer_pack ,
    bool staging_buffer ,
    bool serialize_pack_send ,
    bool wait_all ,
    std::integral_constant<bool,CreateParticles> create_particles)
  {
    UpdateGhostConfig config = {nullptr,comm_tag,gpu_buffer_pack,async_buffer_pack,staging_buffer,serialize_pack_send,wait_all};
    grid_update_ghosts(ldbg,comm,comm_scheme,&grid,domain,grid_cell_values,ghost_comm_buffers,
                       parallel_execution_context,parallel_execution_queue,update_fields,config,create_particles);
  }

}

