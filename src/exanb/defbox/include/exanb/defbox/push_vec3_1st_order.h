#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  struct PushVec3FirstOrderFunctor
  {
    double dt = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz) const
    {
      x += dx * dt;
      y += dy * dt;
      z += dz * dt;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3FirstOrderFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

