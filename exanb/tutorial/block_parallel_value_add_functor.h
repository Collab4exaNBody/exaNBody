#pragma once

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/cuda.h>
#include "array2d.h"

// a specific namespace for our application space
namespace tutorial
{
  using namespace exanb;
    
  // a functor = function applied in parallel
  // it is defined by a class (or struct) with the call operator ()
  struct BlockParallelValueAddFunctor
  {
    Array2DReference m_array;
    double m_value_to_add = 0.0; // value to add

    ONIKA_HOST_DEVICE_FUNC            // works on CPU and GPU
    void operator () (size_t i) const // call operator with i in [0;n[
    {                                 // a whole block (all its threads) execute iteration i
      const size_t cols = m_array.columns();
      ONIKA_CU_BLOCK_SIMD_FOR(size_t, j, 0, cols)   // parallelization among the threads of the current block
      {                                             // for iterations on j in [0;columns[
        m_array[i][j] += m_value_to_add; // each thread executes 0, 1, or multiple iterations of j
      }
    }
  };
}

// specialization of BlockParallelForFunctorTraits, in the onika namespace,
// allows to specify some compile time properties of our functor, like Cuda/HIP compatibility
namespace onika
{
  namespace parallel
  {
    template<>
    struct BlockParallelForFunctorTraits<tutorial::BlockParallelValueAddFunctor>
    {
      static constexpr bool CudaCompatible = true; // or false to prevent the code from being compiled with CUDA
    };
  }
}


