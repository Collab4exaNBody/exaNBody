#pragma once

#include <exanb/particle_neighbors/chunk_neighbors_encode_cell_stream_gpu.h>
#include <exanb/particle_neighbors/chunk_neighbors_gpu_write_accessor.h>

namespace exanb
{
  

  template<class CellT>
  ONIKA_DEVICE_KERNEL_FUNC void chunk_neighbors_gpu_kernel(
    const CellT * cells,
    IJK dims,
//    ssize_t gl,
    double cell_size,
    Vec3d grid_origin,
    IJK grid_offset,
    ReadOnlyAmrGrid amr,
    const ChunkNeighborsConfig config,
    double max_dist,
    unsigned int cs,
    unsigned int cs_log2,
    uint8_t* dev_scratch_mem,
    size_t block_scratch_mem_size,
    GridChunkNeighborsGPUWriteAccessor chunk_neighbors,
    GPUKernelExecutionScratch* scratch,
    bool subcell_compaction = false )
  {  
    if( ONIKA_CU_THREAD_IDX==0 && ONIKA_CU_BLOCK_IDX==0 )
    {
      printf("mem part @%p, base=%p, cap=%u\n", chunk_neighbors.m_fixed_stream_pool, chunk_neighbors.m_fixed_stream_pool->m_base_ptr, chunk_neighbors.m_fixed_stream_pool->m_capacity ) ;
    }
 
      
    // avoid use of compute buffer when possible
    const ssize_t gl = 0; // ignore no ghost option by now
    const IJK dimsNoGL = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
    const uint64_t ncells_no_gl = dimsNoGL.i * dimsNoGL.j * dimsNoGL.k;
    ONIKA_CU_BLOCK_SHARED unsigned int cell_a_no_gl;
    do
    {
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        cell_a_no_gl = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
      }
      ONIKA_CU_BLOCK_SYNC();
      if( cell_a_no_gl < ncells_no_gl )
      {
        const IJK loc_a_no_gl = grid_index_to_ijk( dimsNoGL, cell_a_no_gl );
        const IJK loc_a = { loc_a_no_gl.i+gl , loc_a_no_gl.j+gl , loc_a_no_gl.k+gl };
        const size_t cell_a = grid_ijk_to_index( dims, loc_a );
        const unsigned int n = cells[cell_a].size();
        if( subcell_compaction )
        {
          ChunkNeighborFixedCapacityTemp<true> tmp( dev_scratch_mem + block_scratch_mem_size * ONIKA_CU_BLOCK_IDX , block_scratch_mem_size , n );        
          chunk_neighbors_encode_cell_stream(cells,dims,cell_size,grid_origin,grid_offset,amr,config,max_dist,cs,cs_log2,cell_a,loc_a, chunk_neighbors, tmp );
        }
        else
        {
          ChunkNeighborFixedCapacityTemp<false> tmp( dev_scratch_mem + block_scratch_mem_size * ONIKA_CU_BLOCK_IDX , block_scratch_mem_size , n );        
          chunk_neighbors_encode_cell_stream(cells,dims,cell_size,grid_origin,grid_offset,amr,config,max_dist,cs,cs_log2,cell_a,loc_a, chunk_neighbors, tmp );
        }
      }
    }
    while( cell_a_no_gl < ncells_no_gl );
  }

}

