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

#include <onika/cuda/cuda.h>
#include <onika/soatl/field_tuple.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/grid_fields.h>
#include <onika/yaml/yaml_enum.h>
#include <onika/soatl/field_id_tuple_utils.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>


namespace exanb
{
  namespace UpdateGhostsUtils
  {
    template<typename FieldSetT> struct FieldSetToParticleTuple;
    template<typename... field_ids> struct FieldSetToParticleTuple< FieldSet<field_ids...> > { using type = onika::soatl::FieldTuple<field_ids...>; };
    template<typename FieldSetT> using field_set_to_particle_tuple_t = typename FieldSetToParticleTuple<FieldSetT>::type;

    struct GhostCellParticlesUpdateData
    {
      size_t m_cell_i;
      uint8_t m_particles[0];

      ONIKA_HOST_DEVICE_FUNC
      inline void * particle_data(size_t sizeof_ParticleTuple, size_t idx)
      {
        return (void*) ( m_particles + ( sizeof_ParticleTuple * idx ) );
      }

      ONIKA_HOST_DEVICE_FUNC
      inline const void * particle_data(size_t sizeof_ParticleTuple, size_t idx) const
      {
        return (const void*) ( m_particles + ( sizeof_ParticleTuple * idx ) );
      }
    };

    struct UpdateGhostMpiBuffer
    {
      static constexpr size_t MPI_BUFFER_ALIGN = 4096;
      static_assert( MPI_BUFFER_ALIGN >= onika::memory::GenericHostAllocator::DefaultAlignBytes );
      static constexpr size_t MPI_BUFFER_ALIGN_PAD = MPI_BUFFER_ALIGN - 1;
      static constexpr size_t MPI_BUFFER_ALIGN_MASK = ~ MPI_BUFFER_ALIGN_PAD;

      onika::cuda::CudaDeviceStorage<uint8_t> m_device_buffer;
      std::vector<uint8_t> m_host_buffer;
      onika::cuda::CudaDevice * m_cuda_device = nullptr;

      static inline constexpr std::ptrdiff_t aligned_ptr_diff(std::ptrdiff_t p)
      {
        return ( p + MPI_BUFFER_ALIGN_PAD ) & MPI_BUFFER_ALIGN_MASK;
      }

      static inline constexpr uint8_t* aligned_ptr_base(uint8_t* ptr)
      {
        constexpr uint8_t* zero_ptr = nullptr;
        return zero_ptr + aligned_ptr_diff( ptr - zero_ptr );
      }

      inline uint8_t* data()
      {
        return aligned_ptr_base( ( m_cuda_device != nullptr ) ? m_device_buffer.get() : m_host_buffer.data() );
      }

      inline uint8_t* mpi_buffer()
      {
        return aligned_ptr_base( m_host_buffer.empty() ? m_device_buffer.get() : m_host_buffer.data() );
      }

      inline size_t size() const // payload size
      {
        size_t sz = 0;
        if( m_device_buffer.m_shared != nullptr ) sz = m_device_buffer.m_shared->m_array_size;
        else sz = m_host_buffer.size();
        assert( m_host_buffer.empty() || m_host_buffer.size() == sz );
        if( sz > 0 )
        {
          assert( sz > MPI_BUFFER_ALIGN );
          sz -= MPI_BUFFER_ALIGN;
        }
        return sz;
      }

      inline void clear() // actually releases resources
      {
        m_device_buffer.reset();
        m_host_buffer.clear();
        m_host_buffer.shrink_to_fit();
      }

      inline void resize(size_t req_sz, bool mpi_staging = false )
      {
        if( req_sz > 0 )
        {
          const size_t alloc_size = req_sz + MPI_BUFFER_ALIGN;
          if( m_cuda_device != nullptr )
          {
            if( m_device_buffer.get() == nullptr || alloc_size != m_device_buffer.m_shared->m_array_size )
            {
              m_device_buffer = onika::cuda::CudaDeviceStorage<uint8_t>::New( *m_cuda_device , alloc_size );
            }
          }
          m_host_buffer.clear();
          if( m_cuda_device == nullptr || mpi_staging )
          {
            m_host_buffer.resize( alloc_size );
          }
          else
          {
            m_host_buffer.shrink_to_fit();
          }
        }
        else
        {
          clear();
        }
      }
    };

    struct UpdateGhostPartnerCommInfo
    {
      size_t buffer_offset = 0;
      size_t buffer_size = 0;
      int request_idx = -1;
    };



    struct UpdateGhostsCommManagerBase
    {
      virtual inline ~UpdateGhostsCommManagerBase() {}
    };

    struct UpdateGhostsScratch
    {
      std::shared_ptr<UpdateGhostsCommManagerBase> m_comm_resources = nullptr;
    };

    template<class PackGhostFunctorT, class UnpackGhostFunctorT>
    struct UpdateGhostsCommManager : public UpdateGhostsCommManagerBase
    {
      using PackGhostFunctor = PackGhostFunctorT;
      using UnpackGhostFunctor = UnpackGhostFunctorT;
      using GridCellValueType = typename GridCellValues::GridCellValueType;

      int64_t m_comm_scheme_uid = -1;

      UpdateGhostMpiBuffer send_buffer;
      UpdateGhostMpiBuffer recv_buffer;

      std::vector<UpdateGhostPartnerCommInfo> partner_comm_info;
      std::vector<PackGhostFunctor> pack_functors;
      std::vector<UnpackGhostFunctor> unpack_functors;
      std::vector< MPI_Request > requests;
      std::vector<int> request_to_partner_idx;
      std::vector<int> pack_functor_lane;
      std::vector<int> unpack_functor_lane;
      std::vector<bool> message_sent;

      int active_requests = 0;
      int total_requests = 0;
      int m_num_procs = 0;

      inline int64_t comm_scheme_uid() const { return m_comm_scheme_uid; }

      inline int num_procs() const { return m_num_procs; }

      inline void initialize_number_of_partners(int np)
      {
        // first nprocs elements are reserved for receives from partners
        // last nprocs elements are reserved for sends to partners
        m_num_procs = np;
        partner_comm_info.assign( 2 * np , UpdateGhostPartnerCommInfo{} );
        pack_functors.assign( np , PackGhostFunctor{} );
        unpack_functors.assign( np , UnpackGhostFunctor{} );
        message_sent.assign( np , false );
        pack_functor_lane.assign( np , onika::parallel::UNDEFINED_EXECUTION_LANE );
        unpack_functor_lane.assign( np , onika::parallel::UNDEFINED_EXECUTION_LANE );
      }

      inline UpdateGhostPartnerCommInfo& recv_info(int p) { return partner_comm_info[p]; }
      inline const UpdateGhostPartnerCommInfo& recv_info(int p) const { return partner_comm_info[p]; }
      inline UpdateGhostPartnerCommInfo& send_info(int p) { return partner_comm_info[num_procs()+p]; }
      inline const UpdateGhostPartnerCommInfo& send_info(int p) const { return partner_comm_info[num_procs()+p]; }

      inline void update_from_comm_scheme(
          int rank
        , const GhostCommunicationScheme& comm_scheme
        , auto & ghost_comm_buffers
        , const auto & cells_accessor
        , GridCellValueType * cell_scalars
        , size_t cell_scalar_components
        , const auto & update_fields
        , const GhostBoundaryModifier& ghost_boundary
        , onika::cuda::CudaDevice * alloc_on_device
        , bool staging_buffer
        , bool concurrent_buffer_pack )
      {
        constexpr size_t sizeof_CellParticlesUpdateData = sizeof(GhostCellParticlesUpdateData);
        const size_t sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );
        constexpr size_t sizeof_GridCellValueType = sizeof(GridCellValueType);

        bool rebuild_functors = false;

        if( alloc_on_device != send_buffer.m_cuda_device )
        {
          send_buffer.clear();
          send_buffer.m_cuda_device = alloc_on_device;
          rebuild_functors = true;
        }

        if( alloc_on_device != recv_buffer.m_cuda_device )
        {
          recv_buffer.clear();
          recv_buffer.m_cuda_device = alloc_on_device;
          rebuild_functors = true;
        }

        const int nprocs = comm_scheme.m_partner.size();

        if( comm_scheme.uid() > comm_scheme_uid() )
        {
          initialize_number_of_partners( nprocs );
          assert( nprocs == num_procs() );
          size_t recv_buffer_offset = 0;
          size_t send_buffer_offset = 0;
          for(int p=0;p<nprocs;p++)
          {
            const size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
            const size_t particles_to_receive = comm_scheme.m_partner[p].m_particles_to_receive;
  #         ifndef NDEBUG
            size_t particles_to_receive_chk = 0;
            for(size_t i=0;i<cells_to_receive;i++)
            {
              particles_to_receive_chk += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles;
            }
            assert( particles_to_receive == particles_to_receive_chk );
  #         endif
            std::ptrdiff_t recv_buffer_size = ( cells_to_receive * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_receive * sizeof_ParticleTuple );
            recv_buffer_size = UpdateGhostMpiBuffer::aligned_ptr_diff(recv_buffer_size);

            const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
            const size_t particles_to_send = comm_scheme.m_partner[p].m_particles_to_send;
  #         ifndef NDEBUG
            size_t particles_to_send_chk = 0;
            for(size_t i=0;i<cells_to_send;i++)
            {
              particles_to_send_chk += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
            }
            assert( particles_to_send == particles_to_send_chk );
  #         endif
            std::ptrdiff_t send_buffer_size = ( cells_to_send * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_send * sizeof_ParticleTuple );
            send_buffer_size = UpdateGhostMpiBuffer::aligned_ptr_diff( send_buffer_size );

            recv_info(p).buffer_offset = recv_buffer_offset;
            recv_info(p).buffer_size = recv_buffer_size;
            recv_buffer_offset += recv_buffer_size;

            send_info(p).buffer_offset = send_buffer_offset;
            send_info(p).buffer_size = send_buffer_size;
            send_buffer_offset += send_buffer_size;
          }

          initialize_requests( rank );

          m_comm_scheme_uid = comm_scheme.generate_uid();
          rebuild_functors = true;
          // lout << "scheme UID "<<comm_scheme.uid()<<" -> updated resource UID "<< m_comm_scheme_uid<<std::endl;
        }
        /* else
        {
          lout << "scheme UID "<<comm_scheme.uid()<<" -> cached resource UID "<< m_comm_scheme_uid<<std::endl;
        } */

        // (re)allocate main packing/unpacking buffers if needed
        if( recvbuf_total_size() > recv_buffer.size() ) { rebuild_functors=true; recv_buffer.resize( recvbuf_total_size() , staging_buffer ); }
        if( sendbuf_total_size() > send_buffer.size() ) { rebuild_functors=true; send_buffer.resize( sendbuf_total_size() , staging_buffer ); }

        if( rebuild_functors )
        {
          int next_lane = 0;
          for(int p=0;p<nprocs;p++)
          {
            pack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
            if( pack_functors[p].ready_for_execution() )
            {
              pack_functor_lane[p] = concurrent_buffer_pack ? ( next_lane++ ) : onika::parallel::DEFAULT_EXECUTION_LANE ;
            }
            unpack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
            if( unpack_functors[p].ready_for_execution() )
            {
              unpack_functor_lane[p] = (p==rank) ? pack_functor_lane[p] : ( concurrent_buffer_pack ? ( next_lane++ ) : onika::parallel::DEFAULT_EXECUTION_LANE );
            }
          }
        }
        else
        {
          for(int p=0;p<nprocs;p++)
          {
            pack_functors[p].update_parameters( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
            unpack_functors[p].update_parameters( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
          }
        }
      }

      // returns the number of mpi sends to perform on packed buffers
      inline size_t start_pack_functors( const auto & peq_func, const auto & pec_func, int rank, bool allow_gpu_exec )
      {
        const int nprocs = num_procs();
        size_t mpi_send_count = 0;
        for(int p=0;p<nprocs;p++)
        {
          if( pack_functors[p].ready_for_execution() )
          {
            const size_t cells_to_send = pack_functors[p].cell_count();
            assert( cells_to_send > 0 );
            onika::parallel::BlockParallelForOptions par_for_opts = {};
            par_for_opts.enable_gpu = allow_gpu_exec ;
            peq_func() << onika::parallel::set_lane(pack_functor_lane[p])
                       << onika::parallel::block_parallel_for( cells_to_send, pack_functors[p], pec_func("send_pack") , par_for_opts );
            if( p != rank ) ++ mpi_send_count;
          }
        }
        peq_func() << onika::parallel::flush;
        return mpi_send_count;
      }

      inline void start_unpack_functor( const auto & peq_func, const auto & pec_func, int p, bool allow_gpu_exec )
      {
        if( unpack_functors[p].ready_for_execution() )
        {
          const size_t cells_to_recv = unpack_functors[p].cell_count();
          assert( cells_to_recv > 0 );
          onika::parallel::BlockParallelForOptions par_for_opts = {};
          par_for_opts.enable_gpu = allow_gpu_exec ;
          peq_func() << onika::parallel::set_lane(unpack_functor_lane[p])
                     << onika::parallel::block_parallel_for( cells_to_recv, unpack_functors[p], pec_func("recv_unpack") , par_for_opts )
                     << onika::parallel::flush;
        }
      }

      inline int start_mpi_receives(MPI_Comm comm, int comm_tag, int rank)
      {
        static constexpr bool FWD = PackGhostFunctor::UpdateDirectionToGhost;
        static_assert( FWD == UnpackGhostFunctor::UpdateDirectionToGhost );
        const int nprocs = num_procs();
        const int partner_idx_start = FWD ? 0 : nprocs;
        const size_t buf_total_size = FWD ? recvbuf_total_size() : sendbuf_total_size();
        int active_recvs = 0;
        for(int p=0;p<nprocs;p++)
        {
          const int partner_idx = partner_idx_start + p;
          auto & partner_info = partner_comm_info[partner_idx];
          if( p!=rank && unpack_functors[p].ready_for_execution() )
          {
            assert( unpack_functors[p].buffer_size() > 0 );
            assert( unpack_functors[p].buffer_size() == partner_info.buffer_size );
            assert( partner_info.request_idx != -1 );
            assert( request_to_partner_idx[partner_info.request_idx] == partner_idx );
            assert( partner_info.buffer_offset + partner_info.buffer_size <= buf_total_size );
            assert( partner_rank_from_request_index(partner_info.request_idx) == p );
            MPI_Irecv( (char*) unpack_functors[p].mpi_buffer(), partner_info.buffer_size, MPI_CHAR, p, comm_tag, comm, request_ptr(partner_info.request_idx) );
            ++ active_recvs;
          }
        }
        return active_recvs;
      }

      inline void resize_received_cells(auto * const cells, const auto & cell_allocator, auto create_particles )
      {
        if constexpr ( create_particles )
        {
          const int nprocs = num_procs();
          for(int p=0;p<nprocs;p++)
          {
            unpack_functors[p].resize_received_cells(cells,cell_allocator);
          }
        }
      }

      inline size_t process_received_buffer(const auto & peq_func, const auto & pec_func, int p, bool allow_gpu_exec )
      {
        start_unpack_functor( peq_func, pec_func, p, allow_gpu_exec );
        return unpack_functors[p].cell_count();
      };

      inline int start_ready_mpi_sends(const auto & peq_func, MPI_Comm comm, int comm_tag, int rank, bool serialized_pack)
      {
        static constexpr bool FWD = PackGhostFunctor::UpdateDirectionToGhost;
        static_assert( FWD == UnpackGhostFunctor::UpdateDirectionToGhost );
        const int nprocs = num_procs();
        const int partner_idx_start = FWD ? nprocs : 0;
        const size_t buf_total_size = FWD ? sendbuf_total_size() : recvbuf_total_size();
        int started_sends = 0;
        for(int p=0;p<nprocs;p++)
        {
          const int partner_idx = partner_idx_start + p;
          auto & partner_info = partner_comm_info[partner_idx];
          if( p!=rank && !message_sent[p] && partner_info.buffer_size > 0 )
          {
            assert( pack_functors[p].buffer_size() == partner_info.buffer_size );
            assert( partner_rank_from_request_index(partner_info.request_idx) == p );
            assert( request_to_partner_idx[partner_info.request_idx] == partner_idx );
            assert( partner_info.buffer_offset + partner_info.buffer_size <= buf_total_size );
            const bool ready = serialized_pack ? true : peq_func().query_status( pack_functor_lane[p] );
            if( ready )
            {
              MPI_Isend( (char*) pack_functors[p].mpi_buffer() , partner_info.buffer_size, MPI_CHAR, p, comm_tag, comm, request_ptr(partner_info.request_idx) );
              message_sent[p] = true;
              ++ started_sends;
            }
          }
        }
        return started_sends;
      }

      inline size_t wait_mpi_messages(const auto & peq_func, const auto & pec_func, int rank, bool wait_all, bool gpu_buffer_pack)
      {
        static constexpr bool FWD = PackGhostFunctor::UpdateDirectionToGhost;
        static_assert( FWD == UnpackGhostFunctor::UpdateDirectionToGhost );
        const int nprocs = num_procs();
        size_t ghost_cells_recv = 0;

        if( wait_all ) { MPI_Waitall( number_of_active_requests() , requests_data() , MPI_STATUS_IGNORE ); }

        while( number_of_active_requests() > 0 )
        {
          int reqidx = MPI_UNDEFINED;
          
          if( wait_all ) { reqidx = number_of_active_requests() - 1; }
          else if( number_of_active_requests() == 1 ) { MPI_Wait(requests_data(),MPI_STATUS_IGNORE); reqidx = 0; }
          else MPI_Waitany( number_of_active_requests() , requests_data() , &reqidx , MPI_STATUS_IGNORE );
          
          if( reqidx != MPI_UNDEFINED )
          {
            if( reqidx<0 || reqidx >= number_of_active_requests() ) { fatal_error() << "bad request index "<<reqidx<<std::endl; }
            const int p = partner_rank_from_request_index(reqidx);
            assert( p >= 0 && p < nprocs );
            const bool is_recv = FWD ? request_index_is_recv(reqidx) : request_index_is_send(reqidx);
            if( is_recv )
            {
              assert( p != rank );
              ghost_cells_recv += process_received_buffer(peq_func, pec_func, p, gpu_buffer_pack);
            }
            deactivate_request( reqidx );
          }
          else { fatal_error() << "Undefined request index returned by MPI_Waitany"<<std::endl; }
        }
        
        return ghost_cells_recv;
      }

      inline uint8_t * mpi_send_buffer()
      {
        return send_buffer.mpi_buffer();
      }

      inline uint8_t * mpi_recv_buffer()
      {
        return recv_buffer.mpi_buffer();
      }

      inline void initialize_requests( int rank )
      {
        const int nprocs = num_procs(); //comm_scheme.m_partner.size();

        requests.assign( 2 * nprocs , MPI_REQUEST_NULL );
        request_to_partner_idx.assign( 2 * nprocs , -1 );
        active_requests = 0;

        // rebuild partner index map from requests indices in request array
        for(int p=0;p<nprocs;p++)
        {
          if( p != rank && recv_info(p).buffer_size > 0 )
          {
            recv_info(p).request_idx = active_requests;
            request_to_partner_idx[active_requests] = p;
            ++ active_requests;
          }
        }
        for(int p=0;p<nprocs;p++)
        {
          if( p != rank && send_info(p).buffer_size > 0 )
          {
            send_info(p).request_idx = active_requests;
            request_to_partner_idx[active_requests] = nprocs + p;
            ++ active_requests;
          }
        }
        total_requests = active_requests;
      }

      inline void free_requests()
      {
        for(auto & req : requests)
        {
          if( req != MPI_REQUEST_NULL )
          {
            MPI_Request_free( & req );
          }
        }
        requests.clear();
      }

      inline void reactivate_requests()
      {
        active_requests = total_requests;
        message_sent.assign( num_procs() , false );
      }

      inline uint8_t* sendbuf_ptr(int p) { return send_buffer.data() + send_info(p).buffer_offset; }
      inline size_t sendbuf_total_size() const { return send_info(num_procs()-1).buffer_offset + send_info(num_procs()-1).buffer_size; }

      inline uint8_t* recvbuf_ptr(int p) { return recv_buffer.data() + recv_info(p).buffer_offset; }
      inline size_t recvbuf_total_size() const { return recv_info(num_procs()-1).buffer_offset + recv_info(num_procs()-1).buffer_size; }

      inline int number_of_requests() const { return total_requests; }
      inline int number_of_active_requests() const { return active_requests; }
      inline MPI_Request* requests_data() { return requests.data(); }
      inline MPI_Request* request_ptr(int req_idx) { return & requests[req_idx]; }
      inline int partner_rank_from_request_index(int req_idx) const
      {
        const int p = request_to_partner_idx[req_idx];
        const int np = num_procs();
        return (p < np) ? p : (p - np);
      }
      inline bool request_index_is_recv(int req_idx) const
      {
        const int np = num_procs();
        return request_to_partner_idx[req_idx] < np;
      }
      inline bool request_index_is_send(int req_idx) const { return ! request_index_is_recv(req_idx); }

      // swap request to end of active requests and decreases active request count
      inline void deactivate_request(int req_idx)
      {
        assert( req_idx < active_requests && active_requests > 0 );
        if( req_idx != (active_requests-1) )
        {
          std::swap( requests[req_idx] , requests[active_requests-1] );
          std::swap( request_to_partner_idx[req_idx] , request_to_partner_idx[active_requests-1] );
          partner_comm_info[request_to_partner_idx[req_idx          ]].request_idx = req_idx;
          partner_comm_info[request_to_partner_idx[active_requests-1]].request_idx = active_requests-1;
        }
        -- active_requests;
      }

      inline ~UpdateGhostsCommManager() override final
      {
        // lout <<"free comm resources UID = "<<m_comm_scheme_uid<<std::endl;
        free_requests();
      }

    };

  } // template utilities used only inside UpdateGhostsNode

}
