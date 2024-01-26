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
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/data_types.h>

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/stl_adaptors.h>

namespace exanb
{

  template<
    class GridT,
    class FieldSetT,
    bool CreateParticles,
    class = AssertGridContainFieldSet<GridT,FieldSetT>
    >
  struct UpdateGhostsNode : public OperatorNode
  {  
    using CellParticles = typename GridT::CellParticles;
    using ParticleFullTuple = typename CellParticles::TupleValueType;
    using ParticleTuple = typename UpdateGhostsUtils::FieldSetToParticleTuple< AddDefaultFields<FieldSetT> >::type;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using UpdateGhostsScratch = UpdateGhostsUtils::UpdateGhostsScratch;
    
    static_assert( ParticleTuple::has_field(field::rx) && ParticleTuple::has_field(field::rx) && ParticleTuple::has_field(field::rx) , "ParticleTuple must contain rx, ry and rz fields" );
    
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
      using PackGhostFunctor = UpdateGhostsUtils::GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple>;
      using UnpackGhostFunctor = UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,ParticleFullTuple,CreateParticles>;
          
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
      if( cell_scalars )
      {
        ldbg << "update ghost cell values with "<< cell_scalar_components << " components"<<std::endl;
      }
      else
      {
        cell_scalar_components = 0;
      }

#     ifndef NDEBUG
      const size_t n_cells = grid->number_of_cells();
#     endif      
      
      // initialize MPI requests for both sends and receives
      std::vector< MPI_Request > requests( 2 * nprocs , MPI_REQUEST_NULL );
      std::vector< int > partner_idx( 2 * nprocs , -1 );
      int total_requests = 0;
      //for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }


      // keep track of buffer pack/unpack progress  loop ............................................  2.494e+04            0.000   0.000         1  83.57% / 100.00%
      auto & send_pack_async   = ghost_comm_buffers->send_pack_async;
      auto & recv_unpack_async = ghost_comm_buffers->recv_unpack_async;

      // ***************** send/receive buffers resize ******************
      ghost_comm_buffers->resize_buffers( comm_scheme, sizeof(CellParticlesUpdateData) , sizeof(ParticleTuple) , sizeof(GridCellValueType) , cell_scalar_components );

      // ***************** send bufer packing start ******************
      std::vector<PackGhostFunctor> m_pack_functors( nprocs , PackGhostFunctor{} );
      uint8_t* send_buf_ptr = ghost_comm_buffers->send_buffer.data();
      std::vector<uint8_t> send_staging;
      int active_send_packs = 0;
      if( *staging_buffer )
      {
        send_staging.resize( ghost_comm_buffers->sendbuf_total_size() );
        send_buf_ptr = send_staging.data();
      }

      for(int p=0;p<nprocs;p++)
      {
        if( ghost_comm_buffers->sendbuf_size(p) > 0 )
        {
          assert( ghost_comm_buffers->send_buffer_offsets[p] + ghost_comm_buffers->sendbuf_size(p) <= ghost_comm_buffers->sendbuf_total_size() );
          const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
          m_pack_functors[p] = PackGhostFunctor{ comm_scheme.m_partner[p].m_sends.data()
                                               , cells
                                               , cell_scalars
                                               , cell_scalar_components
                                               , ghost_comm_buffers->sendbuf_ptr(p)
                                               , ghost_comm_buffers->sendbuf_size(p)
                                               , ( (*staging_buffer) && (p!=rank) ) ? ( send_staging.data() + ghost_comm_buffers->send_buffer_offsets[p] ) : nullptr };
          send_pack_async[p] = parallel_execution_context(p);
          onika::parallel::block_parallel_for( cells_to_send, m_pack_functors[p], send_pack_async[p] , (!CreateParticles) && (*gpu_buffer_pack) , *async_buffer_pack );
          if( p != rank ) { ++ active_send_packs; }
        }
      }

      // number of flying messages
      int active_sends = 0;
      int active_recvs = 0;

      // ***************** async receive start ******************
      uint8_t* recv_buf_ptr = ghost_comm_buffers->recv_buffer.data();
      std::vector<uint8_t> recv_staging;
      if( *staging_buffer )
      {
        recv_staging.resize( ghost_comm_buffers->recvbuf_total_size() );
        recv_buf_ptr = recv_staging.data();
      }
      for(int p=0;p<nprocs;p++)
      {
        if( p!=rank && ghost_comm_buffers->recvbuf_size(p) > 0 )
        {
          assert( ghost_comm_buffers->recv_buffer_offsets[p] + ghost_comm_buffers->recvbuf_size(p) <= ghost_comm_buffers->recvbuf_total_size() );
          ++ active_recvs;
          // recv_buf_ptr + ghost_comm_buffers->recv_buffer_offsets[p]
          partner_idx[ total_requests ] = p;
          MPI_Irecv( (char*) recv_buf_ptr + ghost_comm_buffers->recv_buffer_offsets[p], ghost_comm_buffers->recvbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
          ++ total_requests;
        }
      }

      // ***************** initiate buffer sends ******************
      if( *serialize_pack_send )
      {
        for(int p=0;p<nprocs;p++) { if( send_pack_async[p] != nullptr ) { send_pack_async[p]->wait(); } }
      }
      std::vector<bool> message_sent( nprocs , false );
      while( active_sends < active_send_packs )
      {
        for(int p=0;p<nprocs;p++)
        {
          if( p!=rank && !message_sent[p] && ghost_comm_buffers->sendbuf_size(p) > 0 )
          {
            bool ready = true;
            if( ! (*serialize_pack_send) && send_pack_async[p] != nullptr ) { ready = send_pack_async[p]->queryStatus(); }
            if( ready )
            {
              ++ active_sends;
              partner_idx[ total_requests ] = nprocs+p;
              MPI_Isend( (char*) send_buf_ptr + ghost_comm_buffers->send_buffer_offsets[p] , ghost_comm_buffers->sendbuf_size(p), MPI_CHAR, p, comm_tag, comm, & requests[total_requests] );
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
      auto process_receive_buffer = [&,this](int p)
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
            assert( /*cell_i>=0 &&*/ cell_i<n_cells );
            assert( cells[cell_i].empty() );
            cells[cell_i].resize( n_particles , grid->cell_allocator() );
          }
        }
        
        if( p != rank ) ghost_cells_recv += cells_to_receive;
        else ghost_cells_self += cells_to_receive;

        m_unpack_functors[p] = UnpackGhostFunctor{ comm_scheme.m_partner[p].m_receives.data()
                                          , comm_scheme.m_partner[p].m_receive_offset.data()
                                          , (p!=rank) ? ghost_comm_buffers->recvbuf_ptr(p) : ghost_comm_buffers->sendbuf_ptr(p)
                                          , cells 
                                          , cell_scalar_components 
                                          , cell_scalars
                                          , ghost_comm_buffers->recvbuf_size(p)
                                          , ( (*staging_buffer) && (p!=rank) ) ? ( recv_staging.data() + ghost_comm_buffers->recv_buffer_offsets[p] ) : nullptr };
        recv_unpack_async[p] = parallel_execution_context(p);
        onika::parallel::block_parallel_for( cells_to_receive, m_unpack_functors[p], recv_unpack_async[p] , (!CreateParticles) && (*gpu_buffer_pack) , *async_buffer_pack );                    
      };
      // *** end of packet decoding lamda ***

      // manage loopback communication : decode packet directly from sendbuffer without actually receiving it
      if( ghost_comm_buffers->sendbuf_size(rank) > 0 )
      {
        if( ghost_comm_buffers->sendbuf_size(rank) != ghost_comm_buffers->recvbuf_size(rank) )
        {
          fatal_error() << "UpdateFromGhosts: inconsistent loopback communictation : send="<<ghost_comm_buffers->sendbuf_size(rank)<<" receive="<<ghost_comm_buffers->recvbuf_size(rank)<<std::endl;
        }
        ldbg << "UpdateGhosts: loopback buffer size="<<ghost_comm_buffers->sendbuf_size(rank)<<std::endl;
        if( ! (*serialize_pack_send) && send_pack_async[rank] != nullptr ) { send_pack_async[rank]->wait(); }
        process_receive_buffer(rank);
      }

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
        ldbg << "UpdateGhosts: "<< active_sends << " SENDS, "<<active_recvs<<" RECVS, (total "<<total_requests<<") :" ;
        for(int i=0;i<total_requests;i++)
        {
          const int p = partner_idx[i];
          ldbg << " REQ"<< i << "="<< ( (p < nprocs) ? "RECV-P" : "SEND-P" ) << ( (p < nprocs) ? p : (p - nprocs) );
        }
        ldbg << std::endl;

        if( *wait_all )
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
        if( recv_unpack_async[p] != nullptr ) { recv_unpack_async[p]->wait(); }
      }

      if( CreateParticles )
      {
        grid->rebuild_particle_offsets();
      }

      ldbg << "--- end update_ghosts : received "<<ghost_cells_recv<<" cells and loopbacked "<<ghost_cells_self<<" cells"<< std::endl;
     }

  };

}

