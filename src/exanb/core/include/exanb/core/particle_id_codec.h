#pragma once

#include <cstdlib>
#include <cassert>
#include <cstdint>

#include <onika/cuda/cuda.h> 

namespace exanb
{
  
  // type is encoded using 7 bits, up to 128 different particle types
  // cell index is encoded using 36 bits (supports up to 68.10^9 cells)
  // particle index within cell is using with 21 bits (supports up to 2.10^6 particles per cell)
  // total is 64 bits    
  namespace particle_id_codec
  {
    static constexpr unsigned int CELL_INDEX_BITS = 36;
    static constexpr unsigned int PARTICLE_INDEX_BITS = 21;
    static constexpr unsigned int PARTICLE_TYPE_BITS = 7;
    
    static constexpr size_t MAX_CELL_INDEX = (1ull<<CELL_INDEX_BITS) - 1;
    static constexpr size_t MAX_PARTICLE_INDEX = (1ull<<PARTICLE_INDEX_BITS) - 1;
    static constexpr size_t MAX_PARTICLE_TYPE = (1ull<<PARTICLE_TYPE_BITS) - 1;
  }
  
  // encode particle id from its cell, local index in cell and type
  inline uint64_t encode_cell_particle(size_t cell_i, size_t particle_i, unsigned int particle_type=0 )
  {
    using namespace particle_id_codec;
    assert( (CELL_INDEX_BITS+PARTICLE_INDEX_BITS+PARTICLE_TYPE_BITS) == 64 );
    assert( cell_i <= MAX_CELL_INDEX );
    assert( particle_i <= MAX_PARTICLE_INDEX );
    assert( particle_type <= MAX_PARTICLE_TYPE );
    uint64_t id = cell_i;
    id = id << PARTICLE_INDEX_BITS;
    id = id | particle_i;
    id = id << PARTICLE_TYPE_BITS;
    id = id | particle_type;
    return id;
  }
  
  // decode particle cell index, particle in cell index, and type from its encoded id
  ONIKA_HOST_DEVICE_FUNC
  inline void decode_cell_particle(uint64_t id, size_t& cell_i, size_t& particle_i, unsigned int& particle_type)
  {
    using namespace particle_id_codec;
    assert( (CELL_INDEX_BITS+PARTICLE_INDEX_BITS+PARTICLE_TYPE_BITS) == 64 );
    particle_type = id & MAX_PARTICLE_TYPE;
    id = id >> PARTICLE_TYPE_BITS;
    particle_i = id & MAX_PARTICLE_INDEX;
    id = id >> PARTICLE_INDEX_BITS;
    cell_i = id & MAX_CELL_INDEX;
    assert( cell_i <= MAX_CELL_INDEX );
    assert( particle_i <= MAX_PARTICLE_INDEX );
    assert( particle_type <= MAX_PARTICLE_TYPE );
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void decode_cell_particle(uint64_t id, size_t& cell_i, size_t& particle_i)
  {
    unsigned int particle_type = 0;
    decode_cell_particle(id,cell_i,particle_i,particle_type);
  }
}
