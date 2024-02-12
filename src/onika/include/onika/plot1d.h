#pragma once

#include <map>
#include <vector>
#include <string>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>


namespace onika
{
  using Plot1D = onika::memory::CudaMMVector< onika::cuda::pair<double,double> >;

  struct Plot1DSet
  {
    std::map< std::string , Plot1D > m_plots;
    std::map< std::string , std::string > m_captions;
  };

}

