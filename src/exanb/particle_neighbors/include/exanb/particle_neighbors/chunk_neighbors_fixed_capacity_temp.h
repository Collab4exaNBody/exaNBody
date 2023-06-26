#pragma once

#include <onika/cuda/memmove.h>

namespace exanb
{
  

  template<bool SubCellTempCompaction>
  struct ChunkNeighborFixedCapacityTemp
  {

    ONIKA_HOST_DEVICE_FUNC inline ChunkNeighborFixedCapacityTemp(uint8_t* scratch_mem, uint32_t scratch_mem_size, uint32_t npart)
    : smem_part{scratch_mem,scratch_mem_size}
    , n_particles_a(npart)
    , p_a_nbh_status( smem_part.alloc_type<ChunkNeighborsEncodingStatus>( n_particles_a ) , n_particles_a , 0 )
    , cell_a_particle_nbh( smem_part.alloc_type< onika::FixedCapacityVector<uint16_t> >( n_particles_a ) , n_particles_a , n_particles_a )
    {
      if constexpr ( ! SubCellTempCompaction )
      {
        size_t per_part_stream_capacity = (n_particles_a>0) ? smem_part.available_bytes() / ( n_particles_a * sizeof(uint16_t) ) : 0;

        if( per_part_stream_capacity > 0 )
        {
          if( ONIKA_CU_THREAD_IDX == 0 )
          {
            for(unsigned int i=0; i<n_particles_a; i++)
            {
              cell_a_particle_nbh[i] = onika::FixedCapacityVector<uint16_t>( smem_part.alloc_type<uint16_t>( per_part_stream_capacity ) , per_part_stream_capacity );
              cell_a_particle_nbh[i].assign( 1 , 0 ); // first cell group's chunk counter allocated and set to 0
            }
          }
        }
        else
        {
          for(unsigned int i=ONIKA_CU_THREAD_IDX; i<n_particles_a; i+=ONIKA_CU_BLOCK_SIZE)
          {
            cell_a_particle_nbh[i] = onika::FixedCapacityVector<uint16_t>( nullptr , 0 );
          }
        }
        ONIKA_CU_BLOCK_FENCE();
        ONIKA_CU_BLOCK_SYNC();        
      }
    }
    
    ONIKA_HOST_DEVICE_FUNC inline void begin_sub_cell( unsigned int p_start_a, unsigned int p_end_a )
    {
      if constexpr ( SubCellTempCompaction )
      {
        auto smem_part_cpy = smem_part;
        const size_t per_part_stream_capacity = smem_part_cpy.available_bytes() / ( (p_end_a-p_start_a) * sizeof(uint16_t) );

        if( per_part_stream_capacity > 0 )
        {
          if( ONIKA_CU_THREAD_IDX == 0 )
          {
            for(unsigned int p_a=p_start_a; p_a<p_end_a; p_a++)
            {
              cell_a_particle_nbh[p_a] = onika::FixedCapacityVector<uint16_t>( smem_part_cpy.alloc_type<uint16_t>( per_part_stream_capacity ) , per_part_stream_capacity );
              cell_a_particle_nbh[p_a].assign( 1 , 0 ); // first cell group's chunk counter allocated and set to 0
            }
          }
        }
        else
        {
          for( unsigned int p_a = p_start_a+ONIKA_CU_THREAD_IDX ; p_a<p_end_a ; p_a+=ONIKA_CU_BLOCK_SIZE )
          {
            cell_a_particle_nbh[p_a] = onika::FixedCapacityVector<uint16_t>( 0 , 0 );
          }
        }

      }

      assert( p_a_nbh_status.capacity() >= (p_end_a - p_start_a) );
      
      // p_a_nbh_status.assign( p_end_a-p_start_a ,  );
      if( ONIKA_CU_THREAD_IDX == 0 ) { p_a_nbh_status.resize( p_end_a - p_start_a ); }
      ONIKA_CU_BLOCK_FENCE();
      ONIKA_CU_BLOCK_SYNC();
      
      assert( p_a_nbh_status.size() >= (p_end_a - p_start_a) );

      // ==> BLOCK PARALLEL FOR HERE <==
      for(unsigned int i=ONIKA_CU_THREAD_IDX; i < (p_end_a - p_start_a) ; i+=ONIKA_CU_BLOCK_SIZE )
      {
        p_a_nbh_status[i] = ChunkNeighborsEncodingStatus{ { 0 , 0 } , onika::cuda::numeric_limits<unsigned int>::max };
      }
      ONIKA_CU_BLOCK_FENCE();
      ONIKA_CU_BLOCK_SYNC();
      
      // NO need for block synchronize, because threads will access the same set of indices in p_a_nbh_status and cell_a_particle_nbh
      // ==> block parallelization on particle p_a index
      
      // BUT block synchronize must not deadlock here, since no execution divergence shall be observed here, so a synchronize can be placed
      // to enforce non divergent execution check, even if it's not necessary
    }

    ONIKA_HOST_DEVICE_FUNC inline unsigned int avail_stream_space( unsigned int p_a ) const
    {
      return cell_a_particle_nbh[p_a].available();
    }

    ONIKA_HOST_DEVICE_FUNC inline void end_sub_cell( unsigned int p_start_a, unsigned int p_end_a )
    {
      if constexpr ( SubCellTempCompaction )
      {
        // compacting cell_a_particle_nbh[ p_start_a .. p_end_a ] to maximize remaining space
        if(p_end_a<n_particles_a)
        {
          for(unsigned int p_a=p_start_a; p_a<p_end_a; p_a++)
          {
            ONIKA_CU_BLOCK_SHARED uint16_t* src_ptr;
            ONIKA_CU_BLOCK_SHARED size_t src_size;
            ONIKA_CU_BLOCK_SHARED uint16_t* compacted_ptr;
            
            if( ONIKA_CU_THREAD_IDX == 0 )
            {
              src_ptr = cell_a_particle_nbh[p_a].data();
              src_size = cell_a_particle_nbh[p_a].size();
              compacted_ptr = smem_part.alloc_type<uint16_t>( src_size );
            }
            ONIKA_CU_BLOCK_SYNC();
            
            assert( p_a!=p_start_a || compacted_ptr == src_ptr );
            onika::cuda::cu_block_memmove( compacted_ptr , src_ptr , src_size * sizeof(uint16_t) );
            if( ONIKA_CU_THREAD_IDX == 0 )
            {
              cell_a_particle_nbh[p_a] = onika::FixedCapacityVector<uint16_t>( compacted_ptr , src_size , src_size );
            }
          }
        }
        ONIKA_CU_BLOCK_FENCE();
        ONIKA_CU_BLOCK_SYNC();
      }
    }

    onika::memory::MemoryPartionner smem_part;
    size_t n_particles_a;
    onika::FixedCapacityVector<ChunkNeighborsEncodingStatus> p_a_nbh_status;
    onika::FixedCapacityVector< onika::FixedCapacityVector<uint16_t> > cell_a_particle_nbh;
  };

}

