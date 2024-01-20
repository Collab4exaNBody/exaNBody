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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/field_sets.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>

#include <mpi.h>
#include <exanb/mpi/update_from_ghost_utils.h>
#include <exanb/mpi/data_types.h>
#include <exanb/grid_cell_particles/cell_particle_update_functor.h>

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>

namespace exanb
{

  template<
    class GridT,
    class FieldSetT,
    class UpdateFuncT = UpdateValueAdd,
    class = AssertGridContainFieldSet<GridT,FieldSetT>
    >
  struct UpdateFromGhosts : public OperatorNode
  {  
    using CellParticles = typename GridT::CellParticles;
    using ParticleFullTuple = typename CellParticles::TupleValueType;
    using ParticleTuple = typename UpdateGhostsUtils::FieldSetToParticleTuple< FieldSetT >::type;
    using UpdateGhostsScratch = UpdateGhostsUtils::UpdateGhostsScratch;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using UpdateValueFunctor = UpdateFuncT;
        
    struct CellParticlesUpdateData
    {
      size_t m_cell_i;
      ParticleTuple m_particles[0];
    };
    static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridT                    , grid              , INPUT_OUTPUT);
    ADD_SLOT( GridCellValues           , grid_cell_values  , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( long                     , mpi_tag           , INPUT , 0 );

    ADD_SLOT( bool                     , gpu_buffer_pack   , INPUT , false );
    ADD_SLOT( bool                     , async_buffer_pack , INPUT , false );
    ADD_SLOT( bool                     , staging_buffer    , INPUT , false );
    ADD_SLOT( bool                     , serialize_pack_send , INPUT , false );
    ADD_SLOT( bool                     , wait_all          , INPUT , false );

    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers, PRIVATE );

    // implementing generate_tasks instead of execute allows to launch asynchronous block_parallel_for, even with OpenMP backend
    inline void execute() override final
//    inline void generate_tasks() override final
    {    
      using PackGhostFunctor = UpdateFromGhostsUtils::GhostReceivePackToSendBuffer<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple>;
      using UnpackGhostFunctor = UpdateFromGhostsUtils::GhostSendUnpackFromReceiveBuffer<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,UpdateFuncT>;
      using ParForOpts = onika::parallel::BlockParallelForOptions;
      using onika::parallel::block_parallel_for;

      // prerequisites
      MPI_Comm comm = *mpi;
      GhostCommunicationScheme& comm_scheme = *ghost_comm_scheme;

      int comm_tag = *mpi_tag;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);      

      CellParticles* cells = grid->cells();

      // per cell scalar values, if any
      GridCellValueType* cell_scalars = nullptr;
      unsigned int cell_scalar_components = 0;
      if( grid_cell_values.has_value() )
      {
        cell_scalar_components = grid_cell_values->components();
        if( cell_scalar_components > 0 )
        {
          assert( grid_cell_values->data().size() == grid->number_of_cells() * cell_scalar_components );
          cell_scalars = grid_cell_values->data().data();
        }
      }
      if( cell_scalars != nullptr )
      {
        ldbg << "update from ghost cell values with "<< cell_scalar_components << " components"<<std::endl;
      }
      else
      {
        cell_scalar_components = 0; // just in case pointer is null but number of components is non zero
      }
   
      // reverse order begins here, before the code is the same as in update_ghosts.h
      
      // initialize MPI requests for both sends and receives
      std::vector< MPI_Request > requests( 2 * nprocs , MPI_REQUEST_NULL );
      std::vector< int > partner_idx( 2 * nprocs , -1 );
      int total_requests = 0;
      //for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }

      // ***************** send/receive bufer resize ******************
      ghost_comm_buffers->resize_buffers( comm_scheme, sizeof(CellParticlesUpdateData) , sizeof(ParticleTuple) , sizeof(GridCellValueType) , cell_scalar_components );
      auto & send_pack_async   = ghost_comm_buffers->send_pack_async;
      auto & recv_unpack_async = ghost_comm_buffers->recv_unpack_async;

      assert( send_pack_async.size() == nprocs );
      assert( recv_unpack_async.size() == nprocs );

      // ***************** send bufer packing start ******************
      std::vector<PackGhostFunctor> m_pack_functors( nprocs );
      uint8_t* send_buf_ptr = ghost_comm_buffers->recv_buffer.data();
      std::vector<uint8_t> send_staging;
      int active_send_packs = 0;
      if( *staging_buffer )
      {
        send_staging.resize( ghost_comm_buffers->recvbuf_total_size() );
        send_buf_ptr = send_staging.data();
      }
      for(int p=0;p<nprocs;p++)
      {
        send_pack_async[p] = onika::parallel::ParallelExecutionStreamQueue{};
        if( ghost_comm_buffers->recvbuf_size(p) > 0 )
        {
          if( p != rank ) { ++ active_send_packs; }
          const size_t cells_to_send = comm_scheme.m_partner[p].m_receives.size();
          m_pack_functors[p] = PackGhostFunctor{ comm_scheme.m_partner[p].m_receives.data() 
                                               , comm_scheme.m_partner[p].m_receive_offset.data()
                                               , ghost_comm_buffers->recvbuf_ptr(p)
                                               , cells
                                               , cell_scalar_components
                                               , cell_scalars
                                               , ghost_comm_buffers->recvbuf_size(p)
                                               , ( (*staging_buffer) && (p!=rank) ) ? ( send_staging.data() + ghost_comm_buffers->recv_buffer_offsets[p] ) : nullptr };
          auto parallel_op = block_parallel_for( cells_to_send, m_pack_functors[p], parallel_execution_context() , ParForOpts{ .enable_gpu = *gpu_buffer_pack } );
          if( *async_buffer_pack ) send_pack_async[p] = ( parallel_execution_stream(p) << std::move(parallel_op) );
        }
      }

      // number of flying messages
      int active_sends = 0;
      int active_recvs = 0;

      // ***************** async receive start ******************
      uint8_t* recv_buf_ptr = ghost_comm_buffers->send_buffer.data();
      std::vector<uint8_t> recv_staging;
      if( *staging_buffer )
      {
        recv_staging.resize( ghost_comm_buffers->sendbuf_total_size() );
        recv_buf_ptr = recv_staging.data();
      }
      for(int p=0;p<nprocs;p++)
      {
        if( p!=rank && ghost_comm_buffers->sendbuf_size(p) > 0 )
        {
          ++ active_recvs;
          partner_idx[ total_requests ] = p;
          MPI_Irecv( (char*) recv_buf_ptr + ghost_comm_buffers->send_buffer_offsets[p], ghost_comm_buffers->sendbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
          //ldbg << "async recv #"<<total_requests<<" partner #"<<p << std::endl;
          ++ total_requests;          
        }
      }

      /*** optional synchronization : wait for all send buffers to be packed before moving on ***/
      if( *serialize_pack_send )
      {
        for(int p=0;p<nprocs;p++) { send_pack_async[p].wait(); }
      }

      // ***************** initiate buffer sends ******************
      std::vector<bool> message_sent( nprocs , false );
      while( active_sends < active_send_packs )
      {
        for(int p=0;p<nprocs;p++)
        {
          if( p!=rank && !message_sent[p] && ghost_comm_buffers->recvbuf_size(p)>0 )
          {
            bool ready = true;
            if( ! (*serialize_pack_send) ) { ready = send_pack_async[p].query_status(); }
            if( ready )
            {
              ++ active_sends;
              partner_idx[ total_requests ] = nprocs+p;
              MPI_Isend( (char*) send_buf_ptr + ghost_comm_buffers->recv_buffer_offsets[p] , ghost_comm_buffers->recvbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
              // ldbg << "async send #"<<total_requests<<" partner #"<<p << std::endl;
              ++ total_requests;          
              message_sent[p] = true;
            }
          }
        }
      }
      assert( active_sends == active_send_packs );


      std::vector<UnpackGhostFunctor> unpack_functors( nprocs , UnpackGhostFunctor{} );
      size_t ghost_cells_recv = 0;


      // *** packet decoding process lambda ***
      auto process_receive_buffer = [&,this](int p)
      {
        recv_unpack_async[p] = onika::parallel::ParallelExecutionStreamQueue{};
        const size_t cells_to_receive = comm_scheme.m_partner[p].m_sends.size();
        ghost_cells_recv += cells_to_receive;
        unpack_functors[p] = UnpackGhostFunctor { comm_scheme.m_partner[p].m_sends.data()
                                                , cells
                                                , cell_scalars
                                                , cell_scalar_components
                                                , (p!=rank) ? ghost_comm_buffers->sendbuf_ptr(p) : ghost_comm_buffers->recvbuf_ptr(p)
                                                , ghost_comm_buffers->sendbuf_size(p)
                                                , ( (*staging_buffer) && (p!=rank) ) ? ( recv_staging.data() + ghost_comm_buffers->send_buffer_offsets[p] ) : nullptr
                                                , UpdateValueFunctor{} };
        // = parallel_execution_context(p);
        auto parallel_op = block_parallel_for( cells_to_receive, unpack_functors[p], parallel_execution_context() , ParForOpts{ .enable_gpu = *gpu_buffer_pack } ); 
        if( *async_buffer_pack ) recv_unpack_async[p] = ( parallel_execution_stream(p) << std::move(parallel_op) );
      };
      // *** end of packet decoding lamda ***


      // manage loopback communication : decode packet directly from sendbuffer without actually receiving it
      if( ghost_comm_buffers->recvbuf_size(rank) > 0 )
      {
        if( ghost_comm_buffers->sendbuf_size(rank) != ghost_comm_buffers->recvbuf_size(rank) )
        {
          fatal_error() << "UpdateFromGhosts: inconsistent loopback communictation : send="<<ghost_comm_buffers->recvbuf_size(rank)<<" receive="<<ghost_comm_buffers->sendbuf_size(rank)<<std::endl;
        }
        ldbg << "UpdateFromGhosts: loopback buffer size="<<ghost_comm_buffers->sendbuf_size(rank)<<std::endl;
        if( ! (*serialize_pack_send) ) { send_pack_async[rank].wait(); }
        process_receive_buffer(rank);
      }

      // *** wait for flying messages to arrive ***
      if( *wait_all )
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
        ldbg <<"UpdateFromGhosts: "<< active_sends << " SENDS, "<<active_recvs<<" RECVS, (total "<<total_requests<<") :" ;
        for(int i=0;i<total_requests;i++)
        {
          const int p = partner_idx[i];
          ldbg << " REQ"<< i << "="<< ( (p < nprocs) ? "RECV-P" : "SEND-P" ) << ( (p < nprocs) ? p : (p - nprocs) );
        }
        ldbg << std::endl;

        if( *wait_all )
        {
          reqidx = total_requests-1; // process communication in reverse order
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
          const int p = partner_idx[ reqidx ]; // get the original request index ( [0;nprocs-1] for receives, [nprocs;2*nprocs-1] for sends )
          if( p < nprocs ) // it's a receive
          {
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
          ldbg << "Warning: undefined request returned by MPI_Waitany"<<std::endl;
        }
      }

      for(int p=0;p<nprocs;p++)
      {
        recv_unpack_async[p].wait();
      }
      
      ldbg << "--- end update_from_ghosts : received "<< ghost_cells_recv<<" ghost cells" << std::endl;
     }

  };

}

