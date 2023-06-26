#pragma once

#include <cstdlib>

namespace exanb
{

    struct DefaultNeighborFilterFunc
    {
      inline bool operator () (double d2, double rcut2,size_t,size_t,size_t,size_t) const { return d2 <= rcut2; }
    };

}

