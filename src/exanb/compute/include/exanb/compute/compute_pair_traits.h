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

namespace exanb
{

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ComputePairTraits
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool ComputeBufferCompatible = true;
    static inline constexpr bool BufferLessCompatible = false;
    static inline constexpr bool CudaCompatible = false;
  };

  // traits to configure wether functor needs particle start/stop procedure call with associated context structure
  struct ComputePairParticleContextStart {};
  struct ComputePairParticleContextStop {};
  template<class FuncT> struct ComputePairParticleContextTraits
  {
    static inline constexpr bool HasParticleContextStart = false;    
    static inline constexpr bool HasParticleContextStop = false;    
  };

  template<class FuncT> struct ComputePairDebugTraits
  {
    static inline constexpr void print_func( const FuncT & ) {}
  };

}

