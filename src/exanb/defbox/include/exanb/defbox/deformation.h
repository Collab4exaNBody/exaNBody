#pragma once

#include <exanb/core/basic_types.h>
#include <cmath>

namespace exanb
{

  struct Deformation
  {
    Vec3d m_angles = { M_PI/2, M_PI/2 , M_PI/2 };
    Vec3d m_extension = { 1. , 1. , 1. };
  };

  struct DeformationRange
  {
    Deformation m_min;
    Deformation m_max;
  };

}

