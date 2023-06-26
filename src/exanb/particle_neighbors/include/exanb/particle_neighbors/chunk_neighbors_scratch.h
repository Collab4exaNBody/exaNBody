#pragma once

#include <vector>
#include <cstdlib>
#include <onika/memory/allocator.h>
#include <onika/cuda/device_storage.h>
#include <onika/cuda/stl_adaptors.h>

namespace exanb
{

  struct alignas(8) ChunkNeighborsEncodingStatus
  {
    onika::cuda::pair<uint16_t,uint16_t> last_chunk;
    unsigned int chunk_count_idx;
  };

  struct ChunkNeighborsPerThreadScratchEncoding
  {
    std::vector< std::vector< uint16_t > > cell_a_particle_nbh;
    std::vector< ChunkNeighborsEncodingStatus > encoding_status;
    std::vector< uint8_t > fixed_capacity_scratch;
  };

  struct ChunkNeighborsScratchEncoding
  {
    std::vector< ChunkNeighborsPerThreadScratchEncoding > thread;
    onika::cuda::CudaDeviceStorage<uint8_t> m_cuda_fixed_capacity_scratch;
  };

  struct ChunkNeighborsPerThreadScratchStorage
  {
    std::vector< std::vector< std::pair<uint16_t,uint16_t> > > cell_a_particle_nbh;
    std::vector< uint16_t > encoded_stream;
  };

  struct ChunkNeighborsScratchStorage
  {
    std::vector< ChunkNeighborsPerThreadScratchStorage > thread;
  };

}

