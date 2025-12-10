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

#include <onika/soatl/field_tuple.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <onika/integral_constant.h>

namespace exanb
{

  namespace UpdateFromGhostsUtils
  {

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class UpdateFuncT, class FieldAccTuple>
    struct GhostSendUnpackFromReceiveBuffer
    {
      using FieldIndexSeq = std::make_index_sequence< onika::tuple_size_const_v<FieldAccTuple> >;
      const GhostCellSendScheme * __restrict__ m_sends = nullptr;
      size_t m_cell_count = 0;
      CellsAccessorT m_cells = {};
      GridCellValueType * __restrict__ m_cell_scalars = nullptr;
      size_t m_cell_scalar_components = 0;
      uint8_t* __restrict__ m_data_ptr_base = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * __restrict__ m_staging_buffer_ptr = nullptr;
      GhostBoundaryModifier m_boundary = {};
      UpdateFuncT m_merge_func;
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
          m_data_ptr_base = (p!=rank) ? ghost_comm_buffers.sendbuf_ptr(p) : ghost_comm_buffers.recvbuf_ptr(p);
          m_data_buffer_size = send_info.buffer_size;
          sizeof_ParticleTuple = onika::soatl::field_id_tuple_size_bytes( update_fields );
          m_staging_buffer_ptr = ( staging_buffer && (p!=rank) ) ? ( ghost_comm_buffers.mpi_send_buffer() + send_info.buffer_offset ) : nullptr ;
          m_boundary = ghost_boundary;
          m_merge_func = UpdateFuncT{};
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
          m_merge_func = UpdateFuncT{};
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

      inline uint8_t * mpi_buffer() const { return ( m_staging_buffer_ptr != nullptr ) ? m_staging_buffer_ptr : m_data_ptr_base ; }

      inline size_t cell_count() const { return m_cell_count; }

      inline bool ready_for_execution() const
      {
        return m_data_ptr_base!=nullptr && m_sends!=nullptr;
      }

      template<class fid, class T>
      ONIKA_HOST_DEVICE_FUNC
      inline T apply_field_boundary ( const GhostCellSendScheme& info, const onika::soatl::FieldId<fid>& field, const T& value) const
      {
        using FieldT = onika::soatl::FieldId<fid>;
        const auto& bmin = m_boundary.m_domain_min;
        const auto& bmax = m_boundary.m_domain_max;
        const uint32_t flags_x = info.m_flags >> GhostBoundaryModifier::MASK_SHIFT_X ;
        const uint32_t flags_y = info.m_flags >> GhostBoundaryModifier::MASK_SHIFT_Y ;
        const uint32_t flags_z = info.m_flags >> GhostBoundaryModifier::MASK_SHIFT_Z ;
        if constexpr(std::is_same_v<FieldT,field::_rx>) return GhostBoundaryModifier::apply_coord_modifier(value, bmin.x, bmax.x, flags_x );
        if constexpr(std::is_same_v<FieldT,field::_ry>) return GhostBoundaryModifier::apply_coord_modifier(value, bmin.y, bmax.y, flags_y );
        if constexpr(std::is_same_v<FieldT,field::_rz>) return GhostBoundaryModifier::apply_coord_modifier(value, bmin.z, bmax.z, flags_z );
        if constexpr( std::is_same_v<FieldT,field::_fx>|| std::is_same_v<FieldT,field::_vx> ) return GhostBoundaryModifier::apply_vector_modifier(value, flags_x);
        if constexpr( std::is_same_v<FieldT,field::_fy>|| std::is_same_v<FieldT,field::_vy> ) return GhostBoundaryModifier::apply_vector_modifier(value, flags_y);
        if constexpr( std::is_same_v<FieldT,field::_fz>|| std::is_same_v<FieldT,field::_vz> ) return GhostBoundaryModifier::apply_vector_modifier(value, flags_z);
        return value;
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

      template<class FieldOrSpanT>
      ONIKA_HOST_DEVICE_FUNC
      inline const void * unpack_particle_field( const FieldOrSpanT& _f, const GhostCellSendScheme& info, const void * data_vp, uint64_t cell_i, uint64_t part_i ) const
      {
        if constexpr ( onika::is_span_v<FieldOrSpanT> )
        {
          using FieldT = typename FieldOrSpanT::value_type ;
          using ValueType = typename FieldT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;
          for(const auto & f : _f)
          {
            m_merge_func( m_cells[cell_i][f][part_i]
                        , apply_field_boundary( info, exanb::details::field_id_fom_acc_t<FieldT>{} , *(data++) )
                        , onika::TrueType{} );
          }
          return data;
        }
        else
        {
          using exanb::field_id_fom_acc_v;
          using FieldT = FieldOrSpanT;
          using ValueType = typename FieldT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;        
          m_merge_func( m_cells[cell_i][_f][part_i]
                      , apply_field_boundary( info, exanb::details::field_id_fom_acc_t<FieldT>{} , *(data++) )
                      , onika::TrueType{} );
          return data;
        }
      }

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void unpack_particle_fields( const GhostCellSendScheme& info, const CellParticlesUpdateData * const __restrict__ data, uint64_t cell_i, uint64_t part_i, uint64_t part_j, std::index_sequence<FieldIndex...> ) const
      {
        if constexpr ( sizeof...(FieldIndex) > 0 )
        {
          const void * data_ptr = data->particle_data( sizeof_ParticleTuple , part_j );
          using exanb::field_id_fom_acc_v;
          ( ... , ( data_ptr = unpack_particle_field( m_fields.get(onika::tuple_index_t<FieldIndex>{}) , info , data_ptr , cell_i , part_i ) ) );
        }
      }

      ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
      {
        const size_t particle_offset = m_sends[i].m_send_buffer_offset;
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof_ParticleTuple;
        assert( byte_offset < m_data_buffer_size );
        const uint8_t* data_ptr = m_data_ptr_base + byte_offset; //m_sends[i].m_send_buffer_offset;
        const CellParticlesUpdateData * data = (const CellParticlesUpdateData*) data_ptr;

        assert( data->m_cell_i == m_sends[i].m_partner_cell_i );
        
        const size_t cell_i = m_sends[i].m_cell_i;        
        const uint32_t * const __restrict__ particle_index = onika::cuda::vector_data( m_sends[i].m_particle_i );
        const size_t n_particles = onika::cuda::vector_size( m_sends[i].m_particle_i );
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          // assert( particle_index[j] < m_cells[cell_i].size() );
          unpack_particle_fields( m_sends[i], data, cell_i, particle_index[j], j, FieldIndexSeq{} );
          //m_merge_func( m_cells, cell_i, particle_index[j] , data->m_particles[j] , onika::TrueType{} );
        }
        size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof_ParticleTuple;
        if( m_cell_scalars != nullptr )
        {
          const GridCellValueType* gcv = reinterpret_cast<const GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            m_merge_func( m_cell_scalars[cell_i*m_cell_scalar_components+c] , gcv[c] , onika::TrueType{} );
          }
        }
      }
    };

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class FieldAccTuple >
    struct GhostReceivePackToSendBuffer
    {
      using FieldIndexSeq = std::make_index_sequence< onika::tuple_size_const_v<FieldAccTuple> >;
      const GhostCellReceiveScheme * __restrict__ m_receives = nullptr;
      size_t m_cell_count = 0;
      const uint64_t * __restrict__ m_cell_offset = nullptr; 
      uint8_t * __restrict__ m_data_ptr_base = nullptr;
      CellsAccessorT m_cells = {};
      size_t m_cell_scalar_components = 0;
      const GridCellValueType * __restrict__ m_cell_scalars = nullptr;
      size_t m_data_buffer_size = 0;
      size_t sizeof_ParticleTuple = 0;
      uint8_t * __restrict__ m_staging_buffer_ptr = nullptr;
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
          m_data_ptr_base = ghost_comm_buffers.recvbuf_ptr(p);
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

      inline uint8_t * mpi_buffer() const { return ( m_staging_buffer_ptr != nullptr ) ? m_staging_buffer_ptr : m_data_ptr_base ; }

      inline size_t cell_count() const { return m_cell_count; }

      inline bool ready_for_execution() const
      {
        return m_data_ptr_base!=nullptr && m_receives!=nullptr && m_cell_offset!=nullptr;
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
      inline void * pack_particle_field( const FieldOrSpanT& _f, void* data_vp, uint64_t cell_i, uint64_t part_i ) const
      {
        if constexpr ( onika::is_span_v<FieldOrSpanT> )
        {
          using FieldT = typename FieldOrSpanT::value_type ;
          using ValueType = typename FieldT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;
          for(const auto & f : _f) { * (data++) = m_cells[cell_i][f][part_i]; }
          return data;
        }
        else
        {
          using FieldT = FieldOrSpanT;
          using ValueType = typename FieldT::value_type ;
          ValueType * data = ( ValueType * ) data_vp;
          * (data++) = m_cells[cell_i][_f][part_i];
          return data;
        }
      }

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void pack_particle_fields( CellParticlesUpdateData* data, uint64_t cell_i, uint64_t part_i, uint64_t part_j, std::index_sequence<FieldIndex...> ) const
      {
        if constexpr ( sizeof...(FieldIndex) > 0 )
        {
          void * data_ptr = data->particle_data( sizeof_ParticleTuple , part_j );
          ( ... , ( data_ptr = pack_particle_field( m_fields.get(onika::tuple_index_t<FieldIndex>{}) , data_ptr , cell_i , part_i ) ) );
        }
      }

      ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
      {
        const size_t particle_offset = m_cell_offset[i];
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof_ParticleTuple;
        assert( byte_offset < m_data_buffer_size );
        uint8_t * const __restrict__ data_ptr = m_data_ptr_base + byte_offset;
        CellParticlesUpdateData * const __restrict__ data = (CellParticlesUpdateData*) data_ptr;

        const auto cell_input_it = m_receives[i];
        const auto cell_input = ghost_cell_receive_info(cell_input_it);
        const size_t cell_i = cell_input.m_cell_i;
        
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          data->m_cell_i = cell_i; // this is my cell #, recipient will have to translate it to its own cell #
        }

        const size_t n_particles = cell_input.m_n_particles;
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          pack_particle_fields( data, cell_i, j, j, FieldIndexSeq{} );
          //m_cells[cell_i].read_tuple( j , data->m_particles[j] );
        }
        const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof_ParticleTuple;
        if( m_cell_scalars != nullptr )
        {
          GridCellValueType* gcv = reinterpret_cast<GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            gcv[c] = m_cell_scalars[cell_i*m_cell_scalar_components+c];
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

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class MergeFuncT, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateFromGhostsUtils::GhostSendUnpackFromReceiveBuffer<CellParticles,GridCellValueType,CellParticlesUpdateData,MergeFuncT,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateFromGhostsUtils::GhostReceivePackToSendBuffer<CellParticles,GridCellValueType,CellParticlesUpdateData,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

  }

}

