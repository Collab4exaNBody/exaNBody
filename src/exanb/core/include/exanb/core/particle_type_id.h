#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <string>

namespace exanb
{
  // defines mapping between particle type (or category, or class, etc.) name to type id
   using ParticleTypeMap = std::map< std::string , int64_t >;
}
