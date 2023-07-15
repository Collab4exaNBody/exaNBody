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

    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers, PRIVATE );

    // implementing generate_tasks instead of execute allows to launch asynchronous block_parallel_for, even with OpenMP backend
    //inline void generate_tasks () override final
    inline void execute () override final
    {      
      using PackGhostFunctor = UpdateFromGhostsUtils::GhostReceivePackToSendBufer<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple>;
      using UnpackGhostFunctor = UpdateFromGhostsUtils::GhostSendUnpackFromReceiveBuffer<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,UpdateFuncT>;
          
      // prerequisites
      MPI_Comm comm = *mpi;
      GhostCommunicationScheme& comm_scheme = *ghost_comm_scheme;

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

      int comm_tag = *mpi_tag;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);      
   
      // reverse order begins here, before the code is the same as in update_ghosts.h
      
      // initialize MPI requests for both sends and receives
      size_t total_requests = 2 * nprocs;
      std::vector< MPI_Request > requests( total_requests );
      for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }

      // send and receive buffers
      auto & send_buffer = ghost_comm_buffers->send_buffer;
      auto & receive_buffer = ghost_comm_buffers->receive_buffer;
      auto & send_pack_async = ghost_comm_buffers->send_pack_async;
      auto & recv_unpack_async = ghost_comm_buffers->recv_unpack_async;
      send_buffer.resize(nprocs);
      receive_buffer.resize(nprocs);
      send_pack_async.assign(nprocs , nullptr );
      recv_unpack_async.assign(nprocs , nullptr );

      //size_t total_buffer_size = 0;
      size_t active_sends = 0;
      size_t active_recvs = 0;

      // ***************** send/receive bufer resize ******************
      for(int p=0;p<nprocs;p++)
      {
        receive_buffer[p].clear();
        const size_t cells_to_receive = comm_scheme.m_partner[p].m_sends.size();
        size_t particles_to_receive = 0;
        for(size_t i=0;i<cells_to_receive;i++)
        {
          particles_to_receive += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size(); // ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles;
        }
        size_t receive_size = ( cells_to_receive * sizeof(CellParticlesUpdateData) ) + ( particles_to_receive * sizeof(ParticleTuple) );
        if( cell_scalars != nullptr )
        {
          receive_size += cells_to_receive * sizeof(GridCellValueType) * cell_scalar_components;
        }
        receive_buffer[p].resize( receive_size );
        
        send_buffer[p].clear();
        const size_t cells_to_send = comm_scheme.m_partner[p].m_receives.size();
        size_t particles_to_send = 0;
        for(size_t i=0;i<cells_to_send;i++)
        {
          particles_to_send += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles; //comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
        }
        size_t send_buffer_size = ( cells_to_send * sizeof(CellParticlesUpdateData) ) + ( particles_to_send * sizeof(ParticleTuple) );
        if( cell_scalars != nullptr )
        {
          send_buffer_size += cells_to_send * sizeof(GridCellValueType) * cell_scalar_components;
        }
        send_buffer[p].resize( send_buffer_size );
      }

      // ***************** send bufer packing start ******************
      for(int p=0;p<nprocs;p++)
      {
        if( ! send_buffer[p].empty() )
        {
          const size_t cells_to_send = comm_scheme.m_partner[p].m_receives.size();
          PackGhostFunctor pack_ghost = { comm_scheme.m_partner[p].m_receives.data() 
                                        , comm_scheme.m_partner[p].m_receive_offset.data()
                                        , send_buffer[p].data()
                                        , cells
                                        , cell_scalar_components
                                        , cell_scalars };
          send_pack_async[p] = parallel_execution_context(p);
          onika::parallel::block_parallel_for( cells_to_send, pack_ghost, send_pack_async[p] , *gpu_buffer_pack , *async_buffer_pack );
        }
      }

      // ***************** async receive start ******************
      for(int p=0;p<nprocs;p++)
      {
        if( ! receive_buffer[p].empty() )
        {
          ++ active_recvs;
          MPI_Irecv( (char*) receive_buffer[p].data(), receive_buffer[p].size(), MPI_CHAR, p, comm_tag, comm, & requests[p] );
        }
      }

      // ***************** wait for send buffers to be ready ******************
      for(int p=0;p<nprocs;p++)
      {
        if( ! send_buffer[p].empty() )
        {
          if( send_pack_async[p] != nullptr ) { send_pack_async[p]->wait(); }
          //ldbg << "sending "<<send_buffer_size<<" bytes to P"<<p<<std::endl;
          ++ active_sends;
          MPI_Isend( (char*) send_buffer[p].data() , send_buffer[p].size(), MPI_CHAR, p, comm_tag, comm, & requests[nprocs+p] );
        }
      }

      size_t ghost_particles_recv = 0;
      while( active_sends>0 || active_recvs>0 )
      {
        int reqidx = MPI_UNDEFINED;
        MPI_Status status;
        MPI_Waitany(total_requests,requests.data(),&reqidx,&status);
        if( reqidx != MPI_UNDEFINED )
        {
          if( reqidx < nprocs ) // it's a receive
          {
            int p = reqidx;
            const size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
            UnpackGhostFunctor unpack_ghost = { comm_scheme.m_partner[p].m_sends.data()
                                              , cells
                                              , cell_scalars
                                              , cell_scalar_components
                                              , receive_buffer[p].data()
                                              , UpdateValueFunctor{} };
            recv_unpack_async[p] = parallel_execution_context(p);
            onika::parallel::block_parallel_for( cells_to_receive, unpack_ghost, recv_unpack_async[p] , *gpu_buffer_pack , *async_buffer_pack );            
            
            //assert( data_cur == receive_buffer[p].size() );
            -- active_recvs;
            //ldbg<<"received from P"<<p<<" done, remaining recvs="<<active_recvs<< std::endl;
          }
          else // it's a send
          {
            //int p = reqidx - nprocs;
            -- active_sends;
            //ldbg<<"send to P"<<p<<" done, remaining sends="<<active_sends<< std::endl;
          }
        }
      }      

      for(int p=0;p<nprocs;p++)
      {
        if( recv_unpack_async[p] != nullptr ) { recv_unpack_async[p]->wait(); }
      }
      
      ldbg << "--- end update_from_ghosts : received "<< ghost_particles_recv<<" ghost particles" << std::endl;
     }

  };

}

