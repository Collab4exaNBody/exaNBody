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

  struct ComputePairParticleContextStart {};
  struct ComputePairParticleContextStop {};

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ComputePairTraits;
  
  template<> struct ComputePairTraits<void> // this specialization defines defaults, as void cannot be a callable functor
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool ComputeBufferCompatible      = true;
    static inline constexpr bool BufferLessCompatible         = false;
    static inline constexpr bool CudaCompatible               = false;
    static inline constexpr bool HasParticleContextStart      = false;    
    static inline constexpr bool HasParticleContext           = false;
    static inline constexpr bool HasParticleContextStop       = false;
  };

#define COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT(name,member) \
  template<class T , class = void > struct Test_##member { static inline constexpr bool value = ComputePairTraits<void>::member; }; \
  template<class T> struct Test_##member <T , decltype(void(sizeof(T{}.member))) > { static inline constexpr bool value = T::member; }; \
  template<class FuncT> static inline constexpr bool name = Test_##member < ComputePairTraits<FuncT> >::value

  namespace compute_pair_traits
  {
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( requires_block_synchronous_call_v , RequiresBlockSynchronousCall );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( compute_buffer_compatible_v       , ComputeBufferCompatible );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( buffer_less_compatible_v          , BufferLessCompatible );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( cuda_compatible_v                 , CudaCompatible );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( has_particle_context_v            , HasParticleContext );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( has_particle_context_start_v      , HasParticleContextStart );
    COMPUTE_PAIR_VUFFER_SFINAE_TEST_MEMBER_OR_DEFAULT( has_particle_context_stop_v       , HasParticleContextStop );
  }

  template<class FuncT> struct ComputePairDebugTraits
  {
    static inline constexpr void print_func( const FuncT & ) {}
  };

}

