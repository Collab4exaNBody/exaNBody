#pragma once

#include <cstdlib>
// Application wide constants, shared accros differents apps

#ifndef XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT
#define XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT 512
#endif

namespace exanb
{
  static inline constexpr size_t MAX_PARTICLE_NEIGHBORS = XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT;
}

