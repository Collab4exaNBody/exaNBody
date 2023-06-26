#pragma once

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/memory/allocator.h>

namespace exanb
{

  struct CompactCellParticlePairWeights
  {
    onika::memory::CudaMMVector<uint32_t> m_compressed_weights; // 4x groups of 2 bits, each group of 2 bits : 00=0.0, 01=0.5, 10=1.0, 11=custom(read from m_raw_weights)
    inline void clear()
    {
      m_compressed_weights.clear();
    }
    inline void set_number_of_particles(unsigned int np)
    {
      m_compressed_weights.assign( np , 0 );
    }
    ONIKA_HOST_DEVICE_FUNC inline double pair_weight(unsigned int p, unsigned int n) const
    {
      const auto * w = onika::cuda::vector_data( m_compressed_weights );
      int shift = (n%16) * 2;
      uint32_t x32 = ( w[ w[p] + n/16 ] >> shift ) & 0x03;
      return x32*0.5;
    }
    inline void set_neighbor_weight(unsigned int p, unsigned int n, double x)
    {
      if( n == 0 )
      {
        m_compressed_weights[p] = m_compressed_weights.size();
      }
      uint32_t x32 = 3;
      if( x==0.0 ) { x32=0; }
      else if( x==0.5 ) { x32=1; }
      else if( x==1.0 ) { x32=2; }
      else { std::abort(); }

      int shift = (n%16) * 2;
      if(shift==0) { m_compressed_weights.push_back(x32); }
      else { m_compressed_weights.back() |= x32 << shift; }
    }
  };

  // TODO: remove this type, replace with a using alias
  struct CompactGridPairWeights
  {
    onika::memory::CudaMMVector< CompactCellParticlePairWeights > m_cell_weights;
    inline bool empty() const { return  m_cell_weights.empty(); }
  };

}

