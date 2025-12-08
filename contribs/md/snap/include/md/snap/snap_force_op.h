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

#include <onika/physics/units.h>
#include <onika/physics/constants.h>
#include <exanb/core/concurent_add_contributions.h>
#include <onika/cuda/cuda.h>

#include <cmath>

// tells if we use Complex arithmetic classes or unroll all to scalar expressions
//#define SNAP_AUTOGEN_COMPLEX_MATH 1

#ifndef SNAP_CPU_USE_LOCKS
#define SNAP_CPU_USE_LOCKS false
#endif

#include <md/snap/snap_compute_buffer.h>
#include <md/snap/snap_compute_ui.h>
#include <md/snap/snap_compute_yi.h>
#include <md/snap/snap_compute_duidrj.h>
#include <md/snap/snap_compute_deidrj.h>

#ifdef SNAP_AUTOGEN_COMPLEX_MATH
#include <md/snap/snap_math.h>
#endif

#include <md/snap/snap_force_contrib.h>

namespace md
{
  using namespace exanb;

  // Force operator
  template<class RealT, class RijRealT, class SnapConfParamT, class ComputeBufferT, class CellParticlesT, bool CoopCompute = false>
  struct SnapXSForceOpRealT;

# undef SNAP_COOP_COMPUTE
# define SNAP_COOP_COMPUTE 0

# include <md/snap/snap_force_op_coop.hxx>

# undef SNAP_COOP_COMPUTE
# define SNAP_COOP_COMPUTE 1

# include <md/snap/snap_force_op_coop.hxx>

# undef SNAP_COOP_COMPUTE

//  template<class SnapConfParamT, class ComputeBufferT, class CellParticlesT, bool CoopCompute = false>
//  using SnapXSForceOp = SnapXSForceOpRealT<double,SnapConfParamT,ComputeBufferT,CellParticlesT,CoopCompute>;
}

namespace exanb
{
  template<class RealT, class RijRealT, class SnapConfParamT, class CPBufT, class CellParticlesT, bool CoopCompute >
  struct ComputePairTraits< md::SnapXSForceOpRealT<RealT,RijRealT,SnapConfParamT,CPBufT,CellParticlesT,CoopCompute> >
  {
    static inline constexpr bool ComputeBufferCompatible      = true;
    static inline constexpr bool BlockSharedComputeBuffer     = CoopCompute;
    static inline constexpr bool BufferLessCompatible         = false;
    static inline constexpr bool CudaCompatible               = true;
  };

}
