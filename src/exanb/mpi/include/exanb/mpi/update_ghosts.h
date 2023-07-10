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
  using namespace UpdateGhostsUtils;

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
    using ParticleTuple = typename FieldSetToParticleTuple< AddDefaultFields<FieldSetT> >::type;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    
    static_assert( ParticleTuple::has_field(field::rx) , "ParticleTuple must contain field::rx" );
    static_assert( ParticleTuple::has_field(field::ry) , "ParticleTuple must contain field::ry" );
    static_assert( ParticleTuple::has_field(field::rz) , "ParticleTuple must contain field::rz" );
    
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

    ADD_SLOT( UpdateGhostsScratch      , scratch           , PRIVATE );

    // implementing generate_tasks instead of execute allows to launch asynchronous block_parallel_for, even with OpenMP backend
    inline void generate_tasks () override final
    {      
      using PackGhostFunctor = GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple>;
      using UnpackGhostFunctor = GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,ParticleFullTuple,CreateParticles>;
          
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
      if( cell_scalars )
      {
        ldbg << "update ghost cell values with "<< cell_scalar_components << " components"<<std::endl;
      }

      int comm_tag = *mpi_tag;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);


      // FIXME:
      /*********************** shall be in a separate operator **********************/
      //std::cout<<"cell bytes = "<<comm_scheme.m_cell_bytes<<" , scalar components = "<<cell_scalar_components<< " , particle_bytes = "<< comm_scheme.m_particle_bytes<<" , sizeof(ParticleTuple)="<<sizeof(ParticleTuple)<< std::endl;
      if( comm_scheme.m_cell_bytes != cell_scalar_components*sizeof(GridCellValueType) || comm_scheme.m_particle_bytes != sizeof(ParticleTuple) )
      {
        //std::cout<<"recalcul offsets"<<std::endl;
        for(int p=0;p<nprocs;p++)
        {
          size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
          if( cells_to_send > 0 )
          {
            size_t data_cur = 0;
            for(size_t i=0;i<cells_to_send;i++)
            {
              comm_scheme.m_partner[p].m_sends[i].m_send_buffer_offset = data_cur; 
              size_t n_particles = comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
              //std::cout<<"A: p="<<p<<", i="<<i<<" data_cur1="<<data_cur<<" cell#"<<comm_scheme.m_partner[p].m_sends[i].m_cell_i<<" npart="<<n_particles;
              data_cur += sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
              //std::cout<<" data_cur2="<<data_cur;
              if( cell_scalars != nullptr )
              {
                data_cur += sizeof(GridCellValueType) * cell_scalar_components;
              }
              //std::cout<<" data_cur3="<<data_cur<<std::endl;
            }
          }
          
          size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
          comm_scheme.m_partner[p].m_receive_offset.assign( cells_to_receive , 0 );
          size_t buffer_offset = 0;
          for(size_t i=0;i<cells_to_receive;i++)
          {
            const auto cell_input_it = comm_scheme.m_partner[p].m_receives[i];
            const auto cell_input = ghost_cell_receive_info(cell_input_it);
            //const size_t cell_i = cell_input.m_cell_i;
            //assert( cell_i>=0 && cell_i<n_cells );
            size_t n_particles = cell_input.m_n_particles;
            size_t bytes_to_receive = sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
            if( cell_scalars != nullptr )
            {
              bytes_to_receive += sizeof(GridCellValueType) * cell_scalar_components;
            }
            comm_scheme.m_partner[p].m_receive_offset[i] = buffer_offset;
            buffer_offset += bytes_to_receive;
          }
        }
        
        comm_scheme.m_cell_bytes = cell_scalar_components*sizeof(GridCellValueType);
        comm_scheme.m_particle_bytes = sizeof(ParticleTuple);
      }
      /********************************************************************/
      

#     ifndef NDEBUG
      const size_t n_cells = grid->number_of_cells();
#     endif      
      
      // initialize MPI requests for both sends and receives
      size_t total_requests = 2 * nprocs;
      std::vector< MPI_Request > requests( total_requests );
      for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }

      // send and receive buffers
      auto & send_buffer = scratch->send_buffer;
      auto & receive_buffer = scratch->receive_buffer;
      auto & send_pack_async = scratch->send_pack_async;
      auto & recv_unpack_async = scratch->recv_unpack_async;
      send_buffer.resize(nprocs);
      receive_buffer.resize(nprocs);
      send_pack_async.assign(nprocs , nullptr );
      recv_unpack_async.assign(nprocs , nullptr );

      size_t active_sends = 0;
      size_t active_recvs = 0;
      for(int p=0;p<nprocs;p++)
      {
        // start receive from partner p
        size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
        if( cells_to_receive > 0 )
        {
          size_t particles_to_receive = 0;
          for(size_t i=0;i<cells_to_receive;i++)
          {
            particles_to_receive += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles;
          }
          assert( particles_to_receive > 0 );
          size_t receive_size = ( cells_to_receive * sizeof(CellParticlesUpdateData) ) + ( particles_to_receive * sizeof(ParticleTuple) );
          if( cell_scalars != nullptr )
          {
            receive_size += cells_to_receive * sizeof(GridCellValueType) * cell_scalar_components;
          }
          receive_buffer[p].resize( receive_size );
          //ldbg << "receiving "<<receive_size<<" bytes from P"<<p<<std::endl;
          ++ active_recvs;
          MPI_Irecv( (char*) receive_buffer[p].data(), receive_size, MPI_CHAR, p, comm_tag, comm, & requests[p] );
        }
        
        // start send to partner p
        size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
        if( cells_to_send > 0 )
        {
          size_t particles_to_send = 0;
          for(size_t i=0;i<cells_to_send;i++)
          {
            particles_to_send += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
          }
          assert( particles_to_send > 0 );
          size_t send_buffer_size = ( cells_to_send * sizeof(CellParticlesUpdateData) ) + ( particles_to_send * sizeof(ParticleTuple) );
          if( cell_scalars != nullptr )
          {
            send_buffer_size += cells_to_send * sizeof(GridCellValueType) * cell_scalar_components;
          }
          send_buffer[p].resize( send_buffer_size );

          uint8_t* data_ptr_base = send_buffer[p].data();

          PackGhostFunctor pack_ghost = { comm_scheme.m_partner[p].m_sends.data() , cells , cell_scalars , cell_scalar_components , data_ptr_base };
          send_pack_async[p] = parallel_execution_context(p);
          onika::parallel::block_parallel_for( cells_to_send, pack_ghost, send_pack_async[p] , true, (!CreateParticles) && (*gpu_buffer_pack) , *async_buffer_pack );
        }
        else
        {
          send_buffer[p].clear();
        }
      }

      // send packed buffers when they're ready
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

#           ifndef NDEBUG
            size_t particles_to_receive = 0;
            for(size_t i=0;i<cells_to_receive;i++) { particles_to_receive += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles; }
            ssize_t receive_size = ( cells_to_receive * sizeof(CellParticlesUpdateData) ) + ( particles_to_receive * sizeof(ParticleTuple) );
            if( cell_scalars != nullptr )
            {
              receive_size += cells_to_receive * sizeof(GridCellValueType) * cell_scalar_components;
            }
            int status_count = 0;
            MPI_Get_count(&status,MPI_CHAR,&status_count);
            assert( receive_size == status_count );
#           endif
            
            // re-allocate cells before they receive particles
            if( CreateParticles )
            {
              for( auto cell_input_it : comm_scheme.m_partner[p].m_receives )
              {
                const auto cell_input = ghost_cell_receive_info(cell_input_it);
                const size_t n_particles = cell_input.m_n_particles;
                const size_t cell_i = cell_input.m_cell_i;
                assert( cell_i>=0 && cell_i<n_cells );
                cells[cell_i].resize( n_particles , grid->cell_allocator() );
              }
            }

            UnpackGhostFunctor unpack_ghost = { comm_scheme.m_partner[p].m_receives.data() , comm_scheme.m_partner[p].m_receive_offset.data() , receive_buffer[p].data() , cells , cell_scalar_components , cell_scalars };
            recv_unpack_async[p] = parallel_execution_context(p);
            onika::parallel::block_parallel_for( cells_to_receive, unpack_ghost, recv_unpack_async[p] , true , (!CreateParticles) && (*gpu_buffer_pack) , *async_buffer_pack );            
            
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
      
      if( CreateParticles )
      {
        grid->rebuild_particle_offsets();
      }

      ldbg << "--- end update_ghosts : received "<< ghost_particles_recv<<" ghost particles" << std::endl;
     }

  };

}

