#pragma once

#include <type_traits>
#include <cstdint>
#include <cmath>
#include <exanb/core/basic_types_operators.h>

namespace exanb
{

  // ---- optional transform ------
  struct LinearXForm
  {
    Mat3d m_matrix;
    ONIKA_HOST_DEVICE_FUNC inline Vec3d transformCoord(const Vec3d& x) const noexcept { return m_matrix * x; }
  };

  struct NullXForm
  {
    ONIKA_HOST_DEVICE_FUNC static inline constexpr Vec3d transformCoord(Vec3d x) noexcept { return x; }
  };

}

