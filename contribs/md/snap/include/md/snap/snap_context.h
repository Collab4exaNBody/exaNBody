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

#include <onika/memory/allocator.h>
#include <md/snap/sna.h>

namespace md
{

  using namespace exanb;

  template<class RealT>
  struct SnapXSContextRealT
  {
    using RijRealT = RealT;
    
    SnapExt::SnapConfig m_config;
    SnapInternal::SNARealT<RealT> * sna = nullptr;
//    onika::memory::CudaMMVector<SnapInternal::SnapScratchBuffers> m_scratch;
    onika::memory::CudaMMVector<RealT> m_coefs;    // size = number of bispectrum coefficients
    onika::memory::CudaMMVector<RealT> m_factor;   // size = number of elements
    onika::memory::CudaMMVector<RealT> m_radelem;  // size = number of elements
    onika::memory::CudaMMVector<RealT> m_beta;       // size = number of particles
    onika::memory::CudaMMVector<RealT> m_bispectrum; // size = number of particles
    RijRealT m_rcut = 0.0;
    
  //  inline ~SnapXSContext() { for(auto& s:m_scratch) s.finalize(); }
  };

  using SnapXSContext = SnapXSContextRealT<double>;

}
