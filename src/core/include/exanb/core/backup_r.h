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

#include <vector>
#include <cstdint>
#include <cmath>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>

namespace exanb
{

  ONIKA_HOST_DEVICE_FUNC inline double restore_u32_double(uint32_t x, double o, double r)
  {
    return o + (x*r) / (1ull<<32);
  }

  ONIKA_HOST_DEVICE_FUNC inline uint32_t encode_double_u32(double x, double o, double r)
  {
    using onika::cuda::clamp;
    using onika::cuda::numeric_limits;
    
    double xo = x-o;
    uint32_t xu32_a = clamp( static_cast<int64_t>( (xo*(1ull<<32)) / r ) , static_cast<int64_t>(0) , static_cast<int64_t>( numeric_limits<uint32_t>::max ) );
    uint32_t xu32_b = xu32_a-1;
    uint32_t xu32_c = xu32_a+1;
    double ea = fabs( restore_u32_double(xu32_a,o,r) - x );
    double eb = fabs( restore_u32_double(xu32_b,o,r) - x );
    double ec = fabs( restore_u32_double(xu32_c,o,r) - x );
    if( eb < ea ) { return xu32_b; }
    if( ec < ea ) { return xu32_c; }
    return xu32_a;
  }

  struct PositionBackupData
  {
    using CellPositionBackupVector = onika::memory::CudaMMVector<uint32_t>;
    using PositionBackupVector = onika::memory::CudaMMVector< CellPositionBackupVector >; 
    PositionBackupVector m_data;
    Mat3d m_xform;
  };


}

