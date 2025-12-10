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
    
    template<class T, class FieldT>
    ONIKA_HOST_DEVICE_FUNC
    static inline T apply_particle_boundary( const T& v, const FieldT& f , const GhostBoundaryModifier& boundary , uint32_t flags )
    {
      using fid = typename FieldT::Id;
      using BX = decltype(PositionBackupFieldX);
      using BY = decltype(PositionBackupFieldY);
      using BZ = decltype(PositionBackupFieldZ);

      if constexpr ( std::is_same_v<fid,field::_rx> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.x, boundary.m_domain_max.x , flags >> GhostBoundaryModifier::MASK_SHIFT_X );
      if constexpr ( std::is_same_v<fid,field::_ry> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.y, boundary.m_domain_max.y , flags >> GhostBoundaryModifier::MASK_SHIFT_Y );
      if constexpr ( std::is_same_v<fid,field::_rz> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.z, boundary.m_domain_max.z , flags >> GhostBoundaryModifier::MASK_SHIFT_Z );

      if constexpr ( HAS_POSITION_BACKUP_FIELDS )
      {
        if constexpr ( std::is_same_v<fid,BX> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.x, boundary.m_domain_max.x , flags >> GhostBoundaryModifier::MASK_SHIFT_X );
        if constexpr ( std::is_same_v<fid,BY> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.y, boundary.m_domain_max.y , flags >> GhostBoundaryModifier::MASK_SHIFT_Y );
        if constexpr ( std::is_same_v<fid,BZ> ) return GhostBoundaryModifier::apply_coord_modifier( v , boundary.m_domain_min.z, boundary.m_domain_max.z , flags >> GhostBoundaryModifier::MASK_SHIFT_Z );
      }

      if constexpr ( std::is_same_v<fid,field::_vx> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_X );
      if constexpr ( std::is_same_v<fid,field::_vy> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_Y );
      if constexpr ( std::is_same_v<fid,field::_vz> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_Z );

      if constexpr ( std::is_same_v<fid,field::_fx> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_X );
      if constexpr ( std::is_same_v<fid,field::_fy> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_Y );
      if constexpr ( std::is_same_v<fid,field::_fz> ) return GhostBoundaryModifier::apply_vector_modifier( v , flags >> GhostBoundaryModifier::MASK_SHIFT_Z );
      
      return v;
    }

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
          int pack_next_lane = 0;
          int unpack_next_lane = 0;
          for(int p=0;p<nprocs;p++)
          {
            pack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
            if( pack_functors[p].ready_for_execution() )
            {
              pack_functor_lane[p] = concurrent_buffer_pack ? ( pack_next_lane++ ) : onika::parallel::DEFAULT_EXECUTION_LANE ;
            }

            unpack_functors[p].initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
            if( unpack_functors[p].ready_for_execution() )
            {
              unpack_functor_lane[p] = concurrent_buffer_pack ? ( unpack_next_lane++ ) : onika::parallel::DEFAULT_EXECUTION_LANE ;
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
                     << onika::parallel::block_parallel_for( cells_to_recv, unpack_functors[p], pec_func("recv_unpack") , par_for_opts );
        }
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

      inline void start_send_request(int p)
      {
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

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class FieldAccTuple >
    struct GhostSendPackFunctor
    {
      static constexpr size_t FieldCount = onika::tuple_size_const_v<FieldAccTuple>;
      using FieldIndexSeq = std::make_index_sequence< FieldCount >;
      const GhostCellSendScheme * m_sends = nullptr;
      size_t m_cell_count = 0;
      CellsAccessorT m_cells = {};
      const GridCellValueType * m_cell_scalars = nullptr;
      size_t m_cell_scalar_components = 0;
      uint8_t * m_data_ptr_base = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;
      GhostBoundaryModifier m_boundary = {};
      FieldAccTuple m_fields = {};

      inline void initialize( int rank, int p
                            , const GhostCommunicationScheme& comm_scheme
                            , auto& ghost_comm_buffers
                            , const CellsAccessorT& cells_accessor
                            , GridCellValueType * cell_scalars
                            , size_t cell_scalar_components
                            , const FieldAccTuple& update_fields
                            , const GhostBoundaryModifier& ghost_boundary
                            , bool staging_buffer )
      {
        auto & send_info = ghost_comm_buffers.send_info(p);
        if( send_info.buffer_size > 0 )
        {
          m_sends = comm_scheme.m_partner[p].m_sends.data();
          m_cell_count = comm_scheme.m_partner[p].m_sends.size();
          m_cells = cells_accessor;
          m_cell_scalars = cell_scalars;
          m_cell_scalar_components = cell_scalar_components;
          m_data_ptr_base = ghost_comm_buffers.sendbuf_ptr(p);
          m_data_buffer_size = send_info.buffer_size;
          sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );
          m_staging_buffer_ptr = ( staging_buffer && (p!=rank) ) ? ( ghost_comm_buffers.mpi_send_buffer() + send_info.buffer_offset ) : nullptr ;
          m_boundary = ghost_boundary;
          m_fields = update_fields;
        }
        else
        {
          m_sends = nullptr;
          m_cell_count = 0;
          m_cells = CellsAccessorT{};
          m_cell_scalars = nullptr;
          m_cell_scalar_components = 0;
          m_data_ptr_base = nullptr;
          m_data_buffer_size = 0;
          sizeof_ParticleTuple = 0;
          m_staging_buffer_ptr = nullptr;
          m_boundary = GhostBoundaryModifier{};
          m_fields = FieldAccTuple{};
        }
      }

      inline void update_parameters( int rank, int p
                            , const GhostCommunicationScheme& comm_scheme
                            , auto& ghost_comm_buffers
                            , const CellsAccessorT& cells_accessor
                            , GridCellValueType * cell_scalars
                            , size_t cell_scalar_components
                            , const FieldAccTuple& update_fields
                            , const GhostBoundaryModifier& ghost_boundary
                            , bool staging_buffer )
      {
        initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
      }

      inline size_t cell_count() const { return m_cell_count; }

      inline bool ready_for_execution() const
      {
        return m_data_ptr_base!=nullptr && m_sends!=nullptr ;
      }

      inline void operator () ( onika::parallel::block_parallel_for_gpu_epilog_t , onika::parallel::ParallelExecutionStream* stream ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( m_staging_buffer_ptr, m_data_ptr_base , m_data_buffer_size , stream->m_cu_stream ) );
        }
      }
      
      inline void operator () ( onika::parallel::block_parallel_for_cpu_epilog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          std::memcpy( m_staging_buffer_ptr , m_data_ptr_base , m_data_buffer_size );
        }
      }

      template<class FieldOrSpanT>
      ONIKA_HOST_DEVICE_FUNC
      inline void * pack_particle_field( const FieldOrSpanT& _f, void * data_vp, uint64_t cell_i, uint64_t part_i , uint32_t cell_boundary_flags ) const
      {
        if constexpr ( onika::is_span_v<FieldOrSpanT> )
        {
          using FieldT = typename FieldOrSpanT::value_type ;
          using ValueType = typename FieldT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;
          //const size_t N = _f.size(); auto * f_ptr = _f.data();
          for(const auto & f : _f) { *(data++) = apply_particle_boundary(m_cells[cell_i][f][part_i],f,m_boundary,cell_boundary_flags); }
          return data;
        }
        else
        {
          using ValueType = typename FieldOrSpanT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;
          * (data++) = apply_particle_boundary(m_cells[cell_i][_f][part_i],_f,m_boundary,cell_boundary_flags);
          return data;
        }
      }

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void pack_particle_fields( CellParticlesUpdateData* data, uint64_t cell_i, uint64_t part_i, uint64_t part_j , uint32_t cell_boundary_flags , std::index_sequence<FieldIndex...> ) const
      {
        if constexpr ( sizeof...(FieldIndex) > 0 )
        {
          void * data_ptr = data->particle_data( sizeof_ParticleTuple , part_j );
          ( ... , ( data_ptr = pack_particle_field( m_fields.get(onika::tuple_index_t<FieldIndex>{}) , data_ptr , cell_i, part_i , cell_boundary_flags ) ) );
        }
      }

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( uint64_t i ) const
      {      
        const size_t particle_offset = m_sends[i].m_send_buffer_offset;
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof_ParticleTuple;
        assert( byte_offset < m_data_buffer_size );
        uint8_t* data_ptr = m_data_ptr_base + byte_offset; //m_sends[i].m_send_buffer_offset;
        CellParticlesUpdateData* data = (CellParticlesUpdateData*) data_ptr;

        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          data->m_cell_i = m_sends[i].m_partner_cell_i;
        }        
        const size_t cell_i = m_sends[i].m_cell_i;
        const uint32_t cell_boundary_flags = m_sends[i].m_flags;
                
        const uint32_t * const __restrict__ particle_index = onika::cuda::vector_data( m_sends[i].m_particle_i );
        const size_t n_particles = onika::cuda::vector_size( m_sends[i].m_particle_i );
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          if constexpr ( FieldCount > 0 ) { assert( particle_index[j] < m_cells[cell_i].size() ); }
          // m_cells[ cell_i ].read_tuple( particle_index[j], data->m_particles[j] );
          pack_particle_fields( data, cell_i, particle_index[j] , j , cell_boundary_flags , FieldIndexSeq{} );
          //apply_particle_boundary( data->m_particles[j], m_boundary, cell_boundary_flags );
        }
        if( m_cell_scalars != nullptr )
        {
          const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof_ParticleTuple;
          GridCellValueType* gcv = reinterpret_cast<GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            gcv[c]  = m_cell_scalars[cell_i*m_cell_scalar_components+c];
          }
        }
      }
    };

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, bool CreateParticles, class FieldAccTuple>
    struct GhostReceiveUnpackFunctor
    {
      using FieldIndexSeq = std::make_index_sequence< onika::tuple_size_const_v<FieldAccTuple> >;
      const GhostCellReceiveScheme * m_receives = nullptr;
      size_t m_cell_count = 0;
      const uint64_t * m_cell_offset = nullptr;
      uint8_t * m_data_ptr_base = nullptr;
      CellsAccessorT m_cells = {};
      size_t m_cell_scalar_components = 0;
      GridCellValueType * m_cell_scalars = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;
      FieldAccTuple m_fields = {};

      inline void initialize( int rank, int p
                            , const GhostCommunicationScheme& comm_scheme
                            , auto& ghost_comm_buffers
                            , const CellsAccessorT& cells_accessor
                            , GridCellValueType * cell_scalars
                            , size_t cell_scalar_components
                            , const FieldAccTuple& update_fields
                            , const GhostBoundaryModifier& ghost_boundary
                            , bool staging_buffer )
      {
        auto & recv_info = ghost_comm_buffers.recv_info(p);
        if( recv_info.buffer_size > 0 )
        {
          m_receives = comm_scheme.m_partner[p].m_receives.data();
          m_cell_count = comm_scheme.m_partner[p].m_receives.size();
          m_cell_offset = comm_scheme.m_partner[p].m_receive_offset.data();
          m_data_ptr_base = (p!=rank) ? ghost_comm_buffers.recvbuf_ptr(p) : ghost_comm_buffers.sendbuf_ptr(p) ;
          m_cells = cells_accessor;
          m_cell_scalar_components = cell_scalar_components;
          m_cell_scalars = cell_scalars;
          m_data_buffer_size = recv_info.buffer_size;
          sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );
          m_staging_buffer_ptr = ( staging_buffer && (p!=rank) ) ? ( ghost_comm_buffers.mpi_recv_buffer() + recv_info.buffer_offset ) : nullptr ;
          m_fields = update_fields;
        }
        else
        {
          m_receives = nullptr;
          m_cell_count = 0;
          m_cell_offset = nullptr;
          m_data_ptr_base = nullptr;
          m_cells = CellsAccessorT{};
          m_cell_scalar_components = 0;
          m_cell_scalars = nullptr;
          m_data_buffer_size = 0;
          sizeof_ParticleTuple = 0;
          m_staging_buffer_ptr = nullptr;
          m_fields = FieldAccTuple{};
        }
      }

      inline void update_parameters( int rank, int p
                            , const GhostCommunicationScheme& comm_scheme
                            , auto& ghost_comm_buffers
                            , const CellsAccessorT& cells_accessor
                            , GridCellValueType * cell_scalars
                            , size_t cell_scalar_components
                            , const FieldAccTuple& update_fields
                            , const GhostBoundaryModifier& ghost_boundary
                            , bool staging_buffer )
      {
        initialize( rank, p, comm_scheme, ghost_comm_buffers, cells_accessor, cell_scalars, cell_scalar_components, update_fields, ghost_boundary, staging_buffer );
      }

      inline size_t cell_count() const { return m_cell_count; }

      inline bool ready_for_execution() const
      {
        return m_data_ptr_base!=nullptr && m_receives!=nullptr && m_cell_offset!=nullptr;
      }

      inline void operator () ( onika::parallel::block_parallel_for_gpu_prolog_t , onika::parallel::ParallelExecutionStream* stream ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( m_data_ptr_base , m_staging_buffer_ptr , m_data_buffer_size , stream->m_cu_stream ) );
        }        
      }
    
      inline void operator () ( onika::parallel::block_parallel_for_cpu_prolog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          std::memcpy( m_data_ptr_base , m_staging_buffer_ptr , m_data_buffer_size );
        }
      }

      template<class FieldT>
      ONIKA_HOST_DEVICE_FUNC
      inline const void * unpack_particle_field( const onika::cuda::span<FieldT>& fa, const void * data_vp, uint64_t cell_i, uint64_t part_i ) const
      {
        using ValueType = typename FieldT::value_type ;
        const ValueType * data = ( const ValueType * ) data_vp;
        for(const auto& f : fa) m_cells[cell_i][f][part_i] = * (data++);
        return data;
      }

      template<class FieldT>
      ONIKA_HOST_DEVICE_FUNC
      inline const void * unpack_particle_field( const FieldT& f, const void * data_vp, uint64_t cell_i, uint64_t part_i ) const
      {
        using ValueType = typename FieldT::value_type ;
        const ValueType * data = ( const ValueType * ) data_vp;
        m_cells[cell_i][f][part_i] = * (data++) ;
        return data;
      }

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void unpack_particle_fields( const CellParticlesUpdateData * const __restrict__ data, uint64_t cell_i, uint64_t part_i, std::index_sequence<FieldIndex...> ) const
      {
        if constexpr ( sizeof...(FieldIndex) > 0 )
        {
          const void * data_ptr = data->particle_data( sizeof_ParticleTuple , part_i );
          if constexpr ( CreateParticles ) m_cells[cell_i].set_tuple( part_i , {} ); // zero all fields
          ( ... , ( data_ptr = unpack_particle_field( m_fields.get(onika::tuple_index_t<FieldIndex>{}) , data_ptr , cell_i, part_i ) ) );
        }
      }

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( uint64_t i ) const
      {
        const size_t particle_offset = m_cell_offset[i];
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof_ParticleTuple;
        assert( byte_offset < m_data_buffer_size );
        const uint8_t * const __restrict__ data_ptr = m_data_ptr_base + byte_offset; //m_cell_offset[i];
        const CellParticlesUpdateData * const __restrict__ data = (CellParticlesUpdateData*) data_ptr;

        const auto cell_input_it = m_receives[i];
        const auto cell_input = ghost_cell_receive_info(cell_input_it);
        const size_t cell_i = cell_input.m_cell_i;
        assert( cell_i == data->m_cell_i );
/*
#       ifndef NDEBUG
        const IJK cell_loc = grid_index_to_ijk( m_grid_dims , cell_i );
        assert( inside_grid_shell(m_grid_dims,0,m_ghost_layers,cell_loc) );
        using has_field_rx_t = typename ParticleTuple::template HasField < field::_rx >;
        using has_field_ry_t = typename ParticleTuple::template HasField < field::_ry >;
        using has_field_rz_t = typename ParticleTuple::template HasField < field::_rz >;
        static constexpr bool has_position = has_field_rx_t::value && has_field_ry_t::value && has_field_rz_t::value ;
        const AABB cell_bounds = { m_grid_origin + cell_loc * m_cell_size , m_grid_origin + (cell_loc+1) * m_cell_size };
        const double cell_size_epsilon_sq = ( m_cell_size*1.e-3 ) * ( m_cell_size*1.e-3 );
#       endif
*/
        const size_t n_particles = cell_input.m_n_particles;
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          unpack_particle_fields( data, cell_i , j , FieldIndexSeq{} );
/*
#         ifndef NDEBUG
          if constexpr ( has_position )
          {
            const Vec3d r = { data->m_particles[j][field::rx] , data->m_particles[j][field::ry] , data->m_particles[j][field::rz] };
            assert( is_inside_threshold( cell_bounds , r , cell_size_epsilon_sq ) );
          }
#         endif
*/
        }

        if( m_cell_scalars != nullptr )
        {
          const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof_ParticleTuple;
          const GridCellValueType* gcv = reinterpret_cast<const GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            m_cell_scalars[cell_i*m_cell_scalar_components+c] = gcv[c];
          }
        }
      }
    };

  } // template utilities used only inside UpdateGhostsNode

}


namespace onika
{

  namespace parallel
  {

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, bool CreateParticles, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,CreateParticles,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

  }

}

