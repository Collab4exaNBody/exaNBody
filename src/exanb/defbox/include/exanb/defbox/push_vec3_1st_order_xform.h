#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/basic_types_operators.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  struct PushVec3FirstOrderXFormFunctor
  {
    const Mat3d xform = { 1.0 , 0.0 , 0.0 ,
                          0.0 , 1.0 , 0.0 ,
                          0.0 , 0.0 , 1.0 };
    const double dt = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz) const
    {
      const Vec3d d = xform * Vec3d{dx,dy,dz} ;
      x += d.x * dt;
      y += d.y * dt;
      z += d.z * dt;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3FirstOrderXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

