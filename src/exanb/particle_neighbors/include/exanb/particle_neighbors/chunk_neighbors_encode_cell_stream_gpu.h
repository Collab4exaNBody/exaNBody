#pragma once

#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>

#include <onika/fixed_capacity_vector.h>
#include <onika/memory/memory_partition.h>
#include <onika/cuda/cuda_math.h>

#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_gpu_write_accessor.h>

namespace exanb
{
  

  template<class CellT, bool SubCellTempCompaction >
  ONIKA_HOST_DEVICE_FUNC inline bool chunk_neighbors_encode_cell_stream(
    const CellT * cells,
    IJK dims,
    double cell_size,
    Vec3d grid_origin,
    IJK grid_offset,
    ReadOnlyAmrGrid amr,
    const ChunkNeighborsConfig config,
    double max_dist,
    unsigned int cs,
    unsigned int cs_log2,
    size_t cell_a,
    IJK loc_a, 
    GridChunkNeighborsGPUWriteAccessor chunk_neighbors,
    ChunkNeighborFixedCapacityTemp<SubCellTempCompaction>& tmp ) // final cell global encoded stream for all particles
  {
#   include "chunk_neighbors_encode_cell_stream_impl.hxx"
  }

}

