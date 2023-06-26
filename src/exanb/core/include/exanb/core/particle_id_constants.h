#pragma once

#include <cstdint>
#include <limits>

namespace exanb
{
  static constexpr uint64_t PARTICLE_NO_ID = std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t PARTICLE_MISSING_ID = std::numeric_limits<uint64_t>::max() - 1;
  
  static inline bool is_particle_id_valid( uint64_t id )
  {
    return id!=PARTICLE_NO_ID && id!=PARTICLE_MISSING_ID;
  }
}

