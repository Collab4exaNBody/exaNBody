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
    auto [alloc_on_device,comm_tag,gpu_buffer_pack,async_buffer_pack,staging_buffer,serialize_pack_send,wait_all,device_side_buffer] = config;

    using CellParticles = typename GridT::CellParticles;
    using ParticleFullTuple = typename CellParticles::TupleValueType;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using CellParticlesUpdateData = typename UpdateGhostsUtils::GhostCellParticlesUpdateData;
    
    static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    using CellsAccessorT = std::remove_cv_t< std::remove_reference_t< decltype( gridp->cells_accessor() ) > >;
    using PackGhostFunctor = UpdateGhostsUtils::GhostSendPackFunctor<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,FieldAccTupleT>;
    using UnpackGhostFunctor = UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellsAccessorT,GridCellValueType,CellParticlesUpdateData,CreateParticles,FieldAccTupleT>;
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

    // initialize MPI requests for both sends and receives
    std::vector< MPI_Request > requests( 2 * nprocs , MPI_REQUEST_NULL );
    std::vector< int > partner_idx( 2 * nprocs , -1 );
    int total_requests = 0;
    //for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }

    CellsAccessorT cells_accessor = { nullptr , nullptr };
    if( gridp != nullptr ) cells_accessor = gridp->cells_accessor();

    // ***************** send/receive buffers resize ******************
    ghost_comm_buffers.resize_buffers( comm_scheme, sizeof(CellParticlesUpdateData) , sizeof_ParticleTuple , sizeof(GridCellValueType) , cell_scalar_components, alloc_on_device );
    auto & send_pack_async   = ghost_comm_buffers.send_pack_async;
    auto & recv_unpack_async = ghost_comm_buffers.recv_unpack_async;

    assert( send_pack_async.size() == size_t(nprocs) );
    assert( recv_unpack_async.size() == size_t(nprocs) );

    // ***************** send bufer packing start ******************
    std::vector<PackGhostFunctor> m_pack_functors( nprocs , PackGhostFunctor{} );
    uint8_t* send_buf_ptr = ghost_comm_buffers.send_buffer.data();
    std::vector<uint8_t> send_staging;
    int active_send_packs = 0;
    if( staging_buffer )
    {
      send_staging.resize( ghost_comm_buffers.sendbuf_total_size() );
      send_buf_ptr = send_staging.data();
    }

    ldbg << "update ghost domain : "<<domain << std::endl;

    unsigned int parallel_concurrent_lane = 0;
    for(int p=0;p<nprocs;p++)
    {
      if( ghost_comm_buffers.sendbuf_size(p) > 0 )
      {
        assert( ghost_comm_buffers.send_buffer_offsets[p] + ghost_comm_buffers.sendbuf_size(p) <= ghost_comm_buffers.sendbuf_total_size() );
        if( p != rank ) { ++ active_send_packs; }

        const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
        m_pack_functors[p] = PackGhostFunctor{ comm_scheme.m_partner[p].m_sends.data()
                                             , cells_accessor
                                             , cell_scalars
                                             , cell_scalar_components
                                             , ghost_comm_buffers.sendbuf_ptr(p)
                                             , ghost_comm_buffers.sendbuf_size(p)
                                             , sizeof_ParticleTuple
                                             , ( staging_buffer && (p!=rank) ) ? ( send_staging.data() + ghost_comm_buffers.send_buffer_offsets[p] ) : nullptr
                                             , ghost_boundary
                                             , update_fields };

        ParForOpts par_for_opts = {}; par_for_opts.enable_gpu = (!CreateParticles) && gpu_buffer_pack ;
        auto parallel_op = block_parallel_for( cells_to_send, m_pack_functors[p], parallel_execution_context("send_pack") , par_for_opts );

        if( async_buffer_pack )
        {
          send_pack_async[p] = parallel_concurrent_lane++;
          parallel_execution_queue() << onika::parallel::set_lane(send_pack_async[p]) << std::move(parallel_op) << onika::parallel::flush;
        }
        else
        {
          send_pack_async[p] = onika::parallel::UNDEFINED_EXECUTION_LANE;;
        }
      }
    }

    // number of flying messages
    int active_sends = 0;
    int active_recvs = 0;

    // ***************** async receive start ******************
    uint8_t* recv_buf_ptr = ghost_comm_buffers.recv_buffer.data();
    std::vector<uint8_t> recv_staging;
    if( staging_buffer )
    {
      recv_staging.resize( ghost_comm_buffers.recvbuf_total_size() );
      recv_buf_ptr = recv_staging.data();
    }
    for(int p=0;p<nprocs;p++)
    {
      if( p!=rank && ghost_comm_buffers.recvbuf_size(p) > 0 )
      {
        assert( ghost_comm_buffers.recv_buffer_offsets[p] + ghost_comm_buffers.recvbuf_size(p) <= ghost_comm_buffers.recvbuf_total_size() );
        ++ active_recvs;
        // recv_buf_ptr + ghost_comm_buffers.recv_buffer_offsets[p]
        partner_idx[ total_requests ] = p;
        MPI_Irecv( (char*) recv_buf_ptr + ghost_comm_buffers.recv_buffer_offsets[p], ghost_comm_buffers.recvbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
        ++ total_requests;
      }
    }

    // ***************** initiate buffer sends ******************
    if( serialize_pack_send )
    {
      for(int p=0;p<nprocs;p++) { parallel_execution_queue().wait( send_pack_async[p] ); }
    }
    std::vector<bool> message_sent( nprocs , false );
    while( active_sends < active_send_packs )
    {
      for(int p=0;p<nprocs;p++)
      {
        if( p!=rank && !message_sent[p] && ghost_comm_buffers.sendbuf_size(p) > 0 )
        {
          bool ready = true;
          if( ! serialize_pack_send ) { ready = parallel_execution_queue().query_status( send_pack_async[p] ); }
          if( ready )
          {
            ++ active_sends;
            partner_idx[ total_requests ] = nprocs+p;
            MPI_Isend( (char*) send_buf_ptr + ghost_comm_buffers.send_buffer_offsets[p] , ghost_comm_buffers.sendbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
            ++ total_requests;
            message_sent[p] = true;
          }
        }
      }
    }

    ldbg << "UpdateGhosts : total active requests = "<<total_requests<<std::endl;

    std::vector<UnpackGhostFunctor> m_unpack_functors( nprocs , UnpackGhostFunctor{} );

    size_t ghost_cells_recv=0 , ghost_cells_self=0;

    // *** packet decoding process lambda ***
    auto process_receive_buffer = [&](int p)
    {
      const size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();        
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

      m_unpack_functors[p] = UnpackGhostFunctor{ comm_scheme.m_partner[p].m_receives.data()
                                        , comm_scheme.m_partner[p].m_receive_offset.data()
                                        , (p!=rank) ? ghost_comm_buffers.recvbuf_ptr(p) : ghost_comm_buffers.sendbuf_ptr(p)
                                        , cells_accessor 
                                        , cell_scalar_components 
                                        , cell_scalars
                                        , ghost_comm_buffers.recvbuf_size(p)
                                        , sizeof_ParticleTuple
                                        , ( staging_buffer && (p!=rank) ) ? ( recv_staging.data() + ghost_comm_buffers.recv_buffer_offsets[p] ) : nullptr
                                        , update_fields
#                                       ifndef NDEBUG
                                        , ghost_layers
                                        , grid_dims
                                        , grid_start_position
                                        , cell_size
#                                       endif
                                        };
                                        
      ParForOpts par_for_opts = {}; par_for_opts.enable_gpu = (!CreateParticles) && gpu_buffer_pack ;
      auto parallel_op = block_parallel_for( cells_to_receive, m_unpack_functors[p], parallel_execution_context("recv_unpack") , par_for_opts );

      if( async_buffer_pack )
      {
        recv_unpack_async[p] = parallel_concurrent_lane++;
        parallel_execution_queue() << onika::parallel::set_lane(recv_unpack_async[p]) << std::move(parallel_op) << onika::parallel::flush;
      }
      else
      {
        recv_unpack_async[p] = onika::parallel::UNDEFINED_EXECUTION_LANE;
      }
    };
    // *** end of packet decoding lamda ***

    // manage loopback communication : decode packet directly from sendbuffer without actually receiving it
    if( ghost_comm_buffers.sendbuf_size(rank) > 0 )
    {
      if( ghost_comm_buffers.sendbuf_size(rank) != ghost_comm_buffers.recvbuf_size(rank) )
      {
        fatal_error() << "UpdateFromGhosts: inconsistent loopback communictation : send="<<ghost_comm_buffers.sendbuf_size(rank)<<" receive="<<ghost_comm_buffers.recvbuf_size(rank)<<std::endl;
      }
      ldbg << "UpdateGhosts: loopback buffer size="<<ghost_comm_buffers.sendbuf_size(rank)<<std::endl;
      if( ! serialize_pack_send ) { parallel_execution_queue().wait( send_pack_async[rank] ); }
      process_receive_buffer(rank);
    }

    if( wait_all )
    {
      ldbg << "UpdateFromGhosts: MPI_Waitall, total_requests="<<total_requests<<std::endl;
      MPI_Waitall( total_requests , requests.data() , MPI_STATUS_IGNORE );
      ldbg << "UpdateFromGhosts: MPI_Waitall done"<<std::endl;
    }
    
    while( active_sends>0 || active_recvs>0 )
    {
      int reqidx = MPI_UNDEFINED;

      if( total_requests != ( active_sends + active_recvs ) )
      {
        fatal_error() << "Inconsistent total_active_requests ("<<total_requests<<") != ( "<<active_sends<<" + "<<active_recvs<<" )" <<std::endl;
      }
      ldbg << "UpdateGhosts: "<< active_sends << " SENDS, "<<active_recvs<<" RECVS, (total "<<total_requests<<") :" ;
      for(int i=0;i<total_requests;i++)
      {
        const int p = partner_idx[i];
        ldbg << " REQ"<< i << "="<< ( (p < nprocs) ? "RECV-P" : "SEND-P" ) << ( (p < nprocs) ? p : (p - nprocs) );
      }
      ldbg << std::endl;

      if( wait_all )
      {
        reqidx = total_requests-1; // process communications in reverse order
      }
      else
      {
        if( total_requests == 1 )
        {
          MPI_Wait( requests.data() , MPI_STATUS_IGNORE );
          reqidx = 0;
        }
        else
        {
          MPI_Waitany( total_requests , requests.data() , &reqidx , MPI_STATUS_IGNORE );
        }
      }

      if( reqidx != MPI_UNDEFINED )
      {
        if( reqidx<0 || reqidx >=total_requests )
        {
          fatal_error() << "bad request index "<<reqidx<<std::endl;
        }
        int p = partner_idx[ reqidx ]; // get the original request index ( [0;nprocs-1] for receives, [nprocs;2*nprocs-1] for sends )
        if( p < nprocs ) // it's a receive
        {
          assert( p >= 0 && p < nprocs );
          process_receive_buffer(p);
          -- active_recvs;
        }
        else // it's a send
        {
          //const int p = reqidx - nprocs;
          -- active_sends;
        }

        requests[reqidx] = requests[total_requests-1];
        partner_idx[reqidx] = partner_idx[total_requests-1];
        -- total_requests;
      }
      else
      {
        ldbg << "Warning: undefined request index returned by MPI_Waitany"<<std::endl;
      }
    }      

    for(int p=0;p<nprocs;p++)
    {
      parallel_execution_queue().wait( recv_unpack_async[p] );
    }

    if( CreateParticles )
    {
      gridp->rebuild_particle_offsets();
    }

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

