#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  struct PushVec3SecondOrderFunctor
  {
    const double dt = 1.0;
    const double dt2 = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz, double ddx, double ddy, double ddz) const
    {
      x += dx * dt + ddx * dt2;
      y += dy * dt + ddy * dt2;
      z += dz * dt + ddz * dt2;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3SecondOrderFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

