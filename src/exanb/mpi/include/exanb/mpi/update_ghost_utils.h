#pragma once

#include <onika/soatl/field_tuple.h>
#include <exanb/field_sets.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <exanb/core/gpu_execution_context.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/compute/block_parallel_for.h>

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
      std::vector< onika::memory::CudaMMVector<uint8_t> > send_buffer;
      std::vector< onika::memory::CudaMMVector<uint8_t> > receive_buffer;
      std::vector< GPUKernelExecutionContext* > send_pack_async;
      std::vector< GPUKernelExecutionContext* > recv_unpack_async;      
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple>
    struct GhostSendPackFunctor
    {
      const GhostCellSendScheme * const __restrict__ m_sends = nullptr;
      const CellParticles * const __restrict__ m_cells = nullptr;
      const GridCellValueType * const __restrict__ m_cell_scalars = nullptr;
      const size_t m_cell_scalar_components = 0;
      uint8_t* __restrict__ m_data_ptr_base = nullptr;

      ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
      {
        uint8_t* data_ptr = m_data_ptr_base + m_sends[i].m_send_buffer_offset;
        CellParticlesUpdateData* data = (CellParticlesUpdateData*) data_ptr;
        size_t data_cur = 0;

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
          assert( particle_index[j]>=0 && particle_index[j] < m_cells[cell_i].size() );
          m_cells[ cell_i ].read_tuple( particle_index[j], data->m_particles[j] );
          apply_r_shift( data->m_particles[j] , rx_shift, ry_shift, rz_shift );
        }
        data_cur += sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
        if( m_cell_scalars != nullptr )
        {
          GridCellValueType* gcv = reinterpret_cast<GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            gcv[c]  = m_cell_scalars[cell_i*m_cell_scalar_components+c];
          }
          //data_cur += sizeof(GridCellValueType) * m_cell_scalar_components;
        }
        //data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr + data_cur );
      }
    };

    template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles>
    struct GhostReceiveUnpackFunctor
    {
      const GhostCellReceiveScheme * const __restrict__ m_receives = nullptr;
      const uint64_t * const __restrict__ m_cell_offset = nullptr;
      const uint8_t * const __restrict__ m_data_ptr_base = nullptr;

      CellParticles * const __restrict__ m_cells = nullptr;
      const size_t m_cell_scalar_components = 0;
      GridCellValueType * const __restrict__ m_cell_scalars = nullptr;

      ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
      {
        const uint8_t * const __restrict__ data_ptr = m_data_ptr_base + m_cell_offset[i];
        const CellParticlesUpdateData * const __restrict__ data = (CellParticlesUpdateData*) data_ptr;
        //size_t data_cur = 0;

        const auto cell_input_it = m_receives[i];
        const auto cell_input = ghost_cell_receive_info(cell_input_it);

        //assert( data_cur < receive_buffer[p].size() );
        const size_t cell_i = cell_input.m_cell_i;
        assert( cell_i == data->m_cell_i );
        assert( cell_i>=0 && cell_i<n_cells );
        
        const size_t n_particles = cell_input.m_n_particles;
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , j , 0 , n_particles )
        {
          if constexpr (   CreateParticles ) { m_cells[cell_i].set_tuple  ( j, ParticleFullTuple( data->m_particles[j] ) ); }
          if constexpr ( ! CreateParticles ) { m_cells[cell_i].write_tuple( j, data->m_particles[j]                      ); }
        }
        
        const size_t data_cur = sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
        if( m_cell_scalars != nullptr )
        {
          const GridCellValueType* gcv = reinterpret_cast<const GridCellValueType*>( data_ptr + data_cur );
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , c , 0 , m_cell_scalar_components )
          {
            m_cell_scalars[cell_i*m_cell_scalar_components+c] = gcv[c];
          }
          //data_cur += sizeof(GridCellValueType) * m_cell_scalar_components;
        }              
        //data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr + data_cur );
      }
    };


  } // template utilities used only inside UpdateGhostsNode



  template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple>
  struct BlockParallelForFunctorTraits< UpdateGhostsUtils::GhostSendPackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class CellParticles, class GridCellValueType, class CellParticlesUpdateData, class ParticleTuple, class ParticleFullTuple, bool CreateParticles>
  struct BlockParallelForFunctorTraits< UpdateGhostsUtils::GhostReceiveUnpackFunctor<CellParticles,GridCellValueType,CellParticlesUpdateData,ParticleTuple,ParticleFullTuple,CreateParticles> >
  {
    static inline constexpr bool CudaCompatible = true;
  };


}

