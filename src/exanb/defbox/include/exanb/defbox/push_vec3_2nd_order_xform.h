#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/basic_types_operators.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  struct PushVec3SecondOrderXFormFunctor
  {
    const Mat3d xform = { 1.0 , 0.0 , 0.0 ,
                          0.0 , 1.0 , 0.0 ,
                          0.0 , 0.0 , 1.0 };
    const double dt = 1.0;
    const double dt2 = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz, double ddx, double ddy, double ddz) const
    {
      Vec3d d = xform * ( Vec3d{dx,dy,dz} * dt + Vec3d{ddx,ddy,ddz} * dt2 );
      x += d.x;
      y += d.y;
      z += d.z;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3SecondOrderXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

