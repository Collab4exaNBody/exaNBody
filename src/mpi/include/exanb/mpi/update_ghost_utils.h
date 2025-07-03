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
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/grid_fields.h>
#include <onika/yaml/yaml_enum.h>


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

    struct UpdateGhostsScratch
    {
      static constexpr size_t BUFFER_GUARD_SIZE = 4096;
      std::vector<size_t> send_buffer_offsets;
      std::vector<size_t> recv_buffer_offsets;
      std::vector< int > send_pack_async;
      std::vector< int > recv_unpack_async;    
      onika::memory::CudaMMVector<uint8_t> send_buffer;
      onika::memory::CudaMMVector<uint8_t> recv_buffer;
            
      inline void initialize_partners(int nprocs)
      {
        send_buffer_offsets.assign( nprocs + 1 , 0  );
        recv_buffer_offsets.assign( nprocs + 1 , 0  );
        send_pack_async.resize( nprocs );
        recv_unpack_async.resize( nprocs );
      }

      inline void resize_buffers(const GhostCommunicationScheme& comm_scheme , size_t sizeof_CellParticlesUpdateData , size_t sizeof_ParticleTuple , size_t sizeof_GridCellValueType , size_t cell_scalar_components )
      {
        int nprocs = comm_scheme.m_partner.size();
        initialize_partners( nprocs );
        recv_buffer_offsets[0] = 0;
        send_buffer_offsets[0] = 0;
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
          const size_t receive_size = ( cells_to_receive * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_receive * sizeof_ParticleTuple );
          
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
          const size_t send_buffer_size = ( cells_to_send * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_send * sizeof_ParticleTuple );

          recv_buffer_offsets[p+1] = recv_buffer_offsets[p] + receive_size;
          send_buffer_offsets[p+1] = send_buffer_offsets[p] + send_buffer_size;
        }
      
        if( ( recvbuf_total_size() + BUFFER_GUARD_SIZE ) > recv_buffer.size() )
        {
          recv_buffer.clear();
          recv_buffer.resize( recvbuf_total_size() + BUFFER_GUARD_SIZE );
        }
        if( ( sendbuf_total_size() + BUFFER_GUARD_SIZE ) > send_buffer.size() )
        {
          send_buffer.clear();
          send_buffer.resize( sendbuf_total_size() + BUFFER_GUARD_SIZE );
        }
      }
      
      inline size_t sendbuf_size(int p) const { return send_buffer_offsets[p+1] - send_buffer_offsets[p]; } 
      inline uint8_t* sendbuf_ptr(int p) { return send_buffer.data() + send_buffer_offsets[p]; }
      inline size_t sendbuf_total_size() const { return send_buffer_offsets.back(); } 

      inline size_t recvbuf_size(int p) const { return recv_buffer_offsets[p+1] - recv_buffer_offsets[p]; } 
      inline uint8_t* recvbuf_ptr(int p) { return recv_buffer.data() + recv_buffer_offsets[p]; } 
      inline size_t recvbuf_total_size() const { return recv_buffer_offsets.back(); } 
    };

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class FieldAccTuple >
    struct GhostSendPackFunctor
    {
      static constexpr size_t FieldCount = onika::tuple_size_const_v<FieldAccTuple>;
      using FieldIndexSeq = std::make_index_sequence< FieldCount >;
      const GhostCellSendScheme * m_sends = nullptr;
      CellsAccessorT m_cells = {};
      const GridCellValueType * m_cell_scalars = nullptr;
      size_t m_cell_scalar_components = 0;
      uint8_t * m_data_ptr_base = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;
      GhostBoundaryModifier m_boundary = {};
      FieldAccTuple m_fields = {};

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
      const uint64_t * m_cell_offset = nullptr;
      uint8_t * m_data_ptr_base = nullptr;
      CellsAccessorT m_cells = {};
      size_t m_cell_scalar_components = 0;
      GridCellValueType * m_cell_scalars = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;
      FieldAccTuple m_fields = {};

      // debug members
#     ifndef NDEBUG
      size_t m_ghost_layers = 0;
      IJK m_grid_dims = {0,0,0};
      Vec3d m_grid_origin = {0.,0.,0.};
      double m_cell_size = 0.0;
#     endif
      // ------------

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

