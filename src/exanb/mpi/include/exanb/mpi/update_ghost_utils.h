#pragma once

#include <onika/soatl/field_tuple.h>
#include <exanb/field_sets.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/core/yaml_enum.h>

namespace exanb
{
  namespace UpdateGhostsUtils
  {
    template<typename FieldSetT> struct FieldSetToParticleTuple;
    template<typename... field_ids> struct FieldSetToParticleTuple< FieldSet<field_ids...> > { using type = onika::soatl::FieldTuple<field_ids...>; };
    template<typename FieldSetT> using field_set_to_particle_tuple_t = typename FieldSetToParticleTuple<FieldSetT>::type;

    template<typename TupleT, class fid, class T>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_field_shift( TupleT& t , onika::soatl::FieldId<fid> f, T x )
    {
      static constexpr bool has_field = onika::soatl::field_tuple_has_field_v<TupleT,fid>;
      if constexpr ( has_field ) { t[ f ] += x; }
    }
    
    template<typename TupleT>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_r_shift( TupleT& t , double x, double y, double z )
    {
      apply_field_shift( t , field::rx , x );
      apply_field_shift( t , field::ry , y );
      apply_field_shift( t , field::rz , z );

      if constexpr ( HAS_POSITION_BACKUP_FIELDS )
      {
        apply_field_shift( t , PositionBackupFieldX , x );
        apply_field_shift( t , PositionBackupFieldY , y );
        apply_field_shift( t , PositionBackupFieldZ , z );
      }
    }

    struct UpdateGhostsScratch
    {
      static constexpr size_t BUFFER_GUARD_SIZE = 4096;
      std::vector<size_t> send_buffer_offsets;
      std::vector<size_t> recv_buffer_offsets;
      std::vector< onika::parallel::ParallelExecutionContext* > send_pack_async;
      std::vector< onika::parallel::ParallelExecutionContext* > recv_unpack_async;    
      onika::memory::CudaMMVector<uint8_t> send_buffer;
      onika::memory::CudaMMVector<uint8_t> recv_buffer;
            
      inline void initialize_partners(int nprocs)
      {
        send_buffer_offsets.assign( nprocs + 1 , 0  );
        recv_buffer_offsets.assign( nprocs + 1 , 0  );
        send_pack_async.assign( nprocs , nullptr );
        recv_unpack_async.assign( nprocs , nullptr );
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

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple>
    struct GhostSendPackFunctor
    {
      const GhostCellSendScheme * m_sends = nullptr;
      const CellParticles * m_cells = nullptr;
      const GridCellValueType * m_cell_scalars = nullptr;
      size_t m_cell_scalar_components = 0;
      uint8_t * m_data_ptr_base = nullptr;
      size_t m_data_buffer_size = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( onika::parallel::block_parallel_for_gpu_epilog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          ONIKA_CU_MEMCPY( m_staging_buffer_ptr, m_data_ptr_base , m_data_buffer_size );
        }
      }
      
      inline void operator () ( onika::parallel::block_parallel_for_cpu_epilog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          std::memcpy( m_staging_buffer_ptr , m_data_ptr_base , m_data_buffer_size );
        }
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
        const uint32_t * const __restrict__ particle_index = onika::cuda::vector_data( m_sends[i].m_particle_i );
        const size_t n_particles = onika::cuda::vector_size( m_sends[i].m_particle_i );
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          assert( /*particle_index[j]>=0 &&*/ particle_index[j] < m_cells[cell_i].size() );
          m_cells[ cell_i ].read_tuple( particle_index[j], data->m_particles[j] );
          apply_r_shift( data->m_particles[j] , rx_shift, ry_shift, rz_shift );
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

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles>
    struct GhostReceiveUnpackFunctor
    {
      const GhostCellReceiveScheme * m_receives = nullptr;
      const uint64_t * m_cell_offset = nullptr;
      uint8_t * m_data_ptr_base = nullptr;
      CellParticles * m_cells = nullptr;
      size_t m_cell_scalar_components = 0;
      GridCellValueType * m_cell_scalars = nullptr;
      size_t m_data_buffer_size = 0;
      uint8_t * m_staging_buffer_ptr = nullptr;

      ONIKA_HOST_DEVICE_FUNC
      inline void operator () ( onika::parallel::block_parallel_for_gpu_prolog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          ONIKA_CU_MEMCPY( m_data_ptr_base , m_staging_buffer_ptr , m_data_buffer_size );
        }        
      }
      
      inline void operator () ( onika::parallel::block_parallel_for_cpu_prolog_t ) const
      {
        if( m_data_buffer_size > 0 && m_staging_buffer_ptr != nullptr && m_staging_buffer_ptr != m_data_ptr_base )
        {
          std::memcpy( m_data_ptr_base , m_staging_buffer_ptr , m_data_buffer_size );
        }
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
          if constexpr (   CreateParticles ) { m_cells[cell_i].set_tuple  ( j, ParticleFullTuple( data->m_particles[j] ) ); }
          if constexpr ( ! CreateParticles ) { m_cells[cell_i].write_tuple( j, data->m_particles[j]                      ); }
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

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles>
    struct BlockParallelForFunctorTraits< exanb::UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,ParticleFullTuple,CreateParticles> >
    {
      static inline constexpr bool CudaCompatible = true;
    };

  }

}

