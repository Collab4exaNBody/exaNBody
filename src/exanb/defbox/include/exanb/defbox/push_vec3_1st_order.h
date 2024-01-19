/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
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

