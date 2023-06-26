#pragma once

#include <exanb/core/quaternion.h>
#include <exanb/core/basic_types_def.h>

#include <onika/cuda/cuda.h>

namespace exanb
{

  ONIKA_HOST_DEVICE_FUNC inline Mat3d quaternion_to_matrix(const Quaternion& q)
  {
    Mat3d m;
    m.m11 = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z;
    m.m22 = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z;
    m.m33 = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z;
    m.m21 = 2.0 * (q.x*q.y + q.w*q.z ); 
    m.m12 = 2.0 * (q.x*q.y - q.w*q.z );
    m.m31 = 2.0 * (q.x*q.z - q.w*q.y );
    m.m13 = 2.0 * (q.x*q.z + q.w*q.y );
    m.m32 = 2.0 * (q.y*q.z + q.w*q.x );
    m.m23 = 2.0 * (q.y*q.z - q.w*q.x );
    return m;
  }

}

