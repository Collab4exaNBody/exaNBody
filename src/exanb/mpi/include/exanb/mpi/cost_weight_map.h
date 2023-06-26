#pragma once

#include <cstdlib>
#include <unordered_map>

namespace exanb
{
    // associates an integer particle property to a cost factor
    using CostWeightMap = std::unordered_map<unsigned long,double>;
}


