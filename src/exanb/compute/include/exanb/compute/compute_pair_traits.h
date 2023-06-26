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

