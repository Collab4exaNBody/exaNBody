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
#include <exanb/field_sets.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/yaml_enum.h>


namespace exanb
{
  namespace UpdateGhostsUtils
  {
    template<typename FieldSetT> struct FieldSetToParticleTuple;
    template<typename... field_ids> struct FieldSetToParticleTuple< FieldSet<field_ids...> > { using type = onika::soatl::FieldTuple<field_ids...>; };
    template<typename FieldSetT> using field_set_to_particle_tuple_t = typename FieldSetToParticleTuple<FieldSetT>::type;

    template<typename TupleT, class pos_id, class vel_id, class T>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_field_shift( TupleT& t , onika::soatl::FieldId<pos_id> pos_field, onika::soatl::FieldId<vel_id> vel_field, T x , T rmin, T rmax , double mirror_vel_factor = 1.0 )
    {
      static constexpr bool has_pos_field = onika::soatl::field_tuple_has_field_v<TupleT,pos_id>;
      static constexpr bool has_vel_field = onika::soatl::field_tuple_has_field_v<TupleT,vel_id>;
      if constexpr ( has_pos_field )
      {
        t[pos_field] += x;
        if( t[pos_field] > rmin && t[pos_field] < rmax )
        {
          t[pos_field] = rmax + rmin - t[pos_field];
          if constexpr ( has_vel_field )
          {
            t[vel_field] = t[vel_field] * mirror_vel_factor;
          }
        }
      }
    }
    
    template<typename TupleT>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_r_shift( TupleT& t , double x, double y, double z, double mirror_x_min, double mirror_x_max, double mirror_y_min, double mirror_y_max,double  mirror_z_min, double mirror_z_max, double mirror_vel_factor = 1.0 )
    {
      apply_field_shift( t , field::rx , field::vx , x , mirror_x_min, mirror_x_max , mirror_vel_factor );
      apply_field_shift( t , field::ry , field::vy , y , mirror_y_min, mirror_y_max , mirror_vel_factor );
      apply_field_shift( t , field::rz , field::vz , z , mirror_z_min, mirror_z_max , mirror_vel_factor );
      if constexpr ( HAS_POSITION_BACKUP_FIELDS )
      {
        static constexpr onika::soatl::FieldId<void> no_field = {}; 
        apply_field_shift( t , PositionBackupFieldX , no_field, x , mirror_x_min, mirror_x_max );
        apply_field_shift( t , PositionBackupFieldY , no_field, y , mirror_y_min, mirror_y_max );
        apply_field_shift( t , PositionBackupFieldZ , no_field, z , mirror_z_min, mirror_z_max );
      }
    }

    template<class ParticleTuple>
    struct GhostCellParticlesUpdateData
    {
      size_t m_cell_i;
      ParticleTuple m_particles[0];
    };

    struct UpdateGhostsScratch
    {
      static constexpr size_t BUFFER_GUARD_SIZE = 4096;
      std::vector<size_t> send_buffer_offsets;
      std::vector<size_t> recv_buffer_offsets;
      std::vector< onika::parallel::ParallelExecutionStreamQueue > send_pack_async;
      std::vector< onika::parallel::ParallelExecutionStreamQueue > recv_unpack_async;    
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
          size_t particles_to_receive = 0;
          for(size_t i=0;i<cells_to_receive;i++)
          {
            particles_to_receive += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles;
          }
          const size_t receive_size = ( cells_to_receive * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_receive * sizeof_ParticleTuple );
          
          const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
          size_t particles_to_send = 0;
          for(size_t i=0;i<cells_to_send;i++)
          {
            particles_to_send += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
          }
          const size_t send_buffer_size = ( cells_to_send * ( sizeof_CellParticlesUpdateData + sizeof_GridCellValueType * cell_scalar_components ) ) + ( particles_to_send * sizeof_ParticleTuple );

          recv_buffer_offsets[p+1] = recv_buffer_offsets[p] + receive_size;
          send_buffer_offsets[p+1] = send_buffer_offsets[p] + send_buffer_size;
        }
      
        recv_buffer.clear();
        recv_buffer.resize( recvbuf_total_size() + BUFFER_GUARD_SIZE );
        send_buffer.clear();
        send_buffer.resize( sendbuf_total_size() + BUFFER_GUARD_SIZE );
      }
      
      inline size_t sendbuf_size(int p) const { return send_buffer_offsets[p+1] - send_buffer_offsets[p]; } 
      inline uint8_t* sendbuf_ptr(int p) { return send_buffer.data() + send_buffer_offsets[p]; }
      inline size_t sendbuf_total_size() const { return send_buffer_offsets.back(); } 

      inline size_t recvbuf_size(int p) const { return recv_buffer_offsets[p+1] - recv_buffer_offsets[p]; } 
      inline uint8_t* recvbuf_ptr(int p) { return recv_buffer.data() + recv_buffer_offsets[p]; } 
      inline size_t recvbuf_total_size() const { return recv_buffer_offsets.back(); } 
    };

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple , class FieldAccTuple >
    struct GhostSendPackFunctor
    {
      using FieldIndexSeq = std::make_index_sequence< onika::tuple_size_const_v<FieldAccTuple> >;
      const GhostCellSendScheme * m_sends = nullptr;
      CellsAccessorT m_cells = {};
      const GridCellValueType * m_cell_scalars = nullptr;
      size_t m_cell_scalar_components = 0;
      uint8_t * m_data_ptr_base = nullptr;
      size_t m_data_buffer_size = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;
      Vec3d m_mirror_domain_origin = {0.,0.,0.};
      Vec3d m_mirror_domain_size = {0.,0.,0.}; // if != 0.0 for any direction, then periodic mirroring is applied for this direction
      double m_mirror_speed_factor = 1.0;
      FieldAccTuple m_fields = {};

      struct MirrorRange { double rmin; double rmax; };

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

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void pack_particle_fields( CellParticlesUpdateData* data, uint64_t cell_i, uint64_t i, uint64_t j, std::index_sequence<FieldIndex...> ) const
      {
        data->m_particles[j] = ParticleTuple( m_cells[cell_i][m_fields.get(onika::tuple_index_t<FieldIndex>{})][i] ... );
      }

      ONIKA_HOST_DEVICE_FUNC
      static inline MirrorRange mirror_range(double shift, double size, double origin)
      {
        if( size > 0.0 )
        {
          int r = static_cast<int>( ( shift * 1.5 ) / size );
          if( ( r % 2 ) != 0 )
          {
            return {origin + r * size , origin + (r+1) * size };
          }
        }
        return { 0.0 , 0.0 };
      }

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( uint64_t i ) const
      {
        const size_t particle_offset = m_sends[i].m_send_buffer_offset;
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof(ParticleTuple);
        assert( byte_offset < m_data_buffer_size );
        uint8_t* data_ptr = m_data_ptr_base + byte_offset; //m_sends[i].m_send_buffer_offset;
        CellParticlesUpdateData* data = (CellParticlesUpdateData*) data_ptr;

        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          data->m_cell_i = m_sends[i].m_partner_cell_i;
        }        
        const size_t cell_i = m_sends[i].m_cell_i;
        const double rx_shift = m_sends[i].m_x_shift;
        const double ry_shift = m_sends[i].m_y_shift;
        const double rz_shift = m_sends[i].m_z_shift;
        
        const auto [mirror_x_min,mirror_x_max] = mirror_range( rx_shift, m_mirror_domain_size.x , m_mirror_domain_origin.x );
        const auto [mirror_y_min,mirror_y_max] = mirror_range( ry_shift, m_mirror_domain_size.y , m_mirror_domain_origin.y );
        const auto [mirror_z_min,mirror_z_max] = mirror_range( rz_shift, m_mirror_domain_size.z , m_mirror_domain_origin.z );
        
        const uint32_t * const __restrict__ particle_index = onika::cuda::vector_data( m_sends[i].m_particle_i );
        const size_t n_particles = onika::cuda::vector_size( m_sends[i].m_particle_i );
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          assert( /*particle_index[j]>=0 &&*/ particle_index[j] < m_cells[cell_i].size() );          
          // m_cells[ cell_i ].read_tuple( particle_index[j], data->m_particles[j] );
          pack_particle_fields( data, cell_i, particle_index[j] , j , FieldIndexSeq{} );
          apply_r_shift( data->m_particles[j], rx_shift, ry_shift, rz_shift, mirror_x_min,mirror_x_max, mirror_y_min,mirror_y_max, mirror_z_min,mirror_z_max, m_mirror_speed_factor );
        }
        if( m_cell_scalars != nullptr )
        {
          const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
          GridCellValueType* gcv = reinterpret_cast<GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            gcv[c]  = m_cell_scalars[cell_i*m_cell_scalar_components+c];
          }
        }
      }
    };

    template<class CellsAccessorT, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles, class FieldAccTuple>
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
      uint8_t * m_staging_buffer_ptr = nullptr;
      FieldAccTuple m_fields = {};

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

      template<size_t ... FieldIndex>
      ONIKA_HOST_DEVICE_FUNC
      inline void unpack_particle_fields( const CellParticlesUpdateData * const __restrict__ data, uint64_t cell_i, uint64_t i, std::index_sequence<FieldIndex...> ) const
      {
        using exanb::field_id_fom_acc_v;
        if constexpr ( CreateParticles ) m_cells[cell_i].set_tuple( i , ParticleFullTuple() ); // zero all fields
        ( ... , (
          m_cells[cell_i][ m_fields.get(onika::tuple_index_t<FieldIndex>{}) ][i] = data->m_particles[i][ field_id_fom_acc_v< decltype( m_fields.get_copy(onika::tuple_index_t<FieldIndex>{}) ) > ]
        ) );
      }

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( uint64_t i ) const
      {
        const size_t particle_offset = m_cell_offset[i];
        const size_t byte_offset = i * ( sizeof(CellParticlesUpdateData) + m_cell_scalar_components * sizeof(GridCellValueType) ) + particle_offset * sizeof(ParticleTuple);
        assert( byte_offset < m_data_buffer_size );
        const uint8_t * const __restrict__ data_ptr = m_data_ptr_base + byte_offset; //m_cell_offset[i];
        const CellParticlesUpdateData * const __restrict__ data = (CellParticlesUpdateData*) data_ptr;

        const auto cell_input_it = m_receives[i];
        const auto cell_input = ghost_cell_receive_info(cell_input_it);
        const size_t cell_i = cell_input.m_cell_i;
        assert( cell_i == data->m_cell_i );
        
        const size_t n_particles = cell_input.m_n_particles;
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          unpack_particle_fields( data, cell_i , j , FieldIndexSeq{} );
        }
        
        if( m_cell_scalars != nullptr )
        {
          const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
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

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles, class FieldAccTupleT>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,ParticleFullTuple,CreateParticles,FieldAccTupleT> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

  }

}

