#include <map>
#include <vector>
#include <string>
#include <onika/memory/allocator.h>

namespace onika
{

  using Plot1D = std::CudaMMVector< std::pair<double,double> >;

  struct Plot1DSet
  {
    std::map< std::string , Plot1D > m_plots;
  };

}

