#pragma once

#include <vector>
#include <cstdint>
#include <exanb/core/basic_types.h>

namespace exanb
{
    struct PositionLongTermBackup
    {
      std::vector< uint64_t > m_ids;
      std::vector< Vec3d > m_positions;
      std::vector< Vec3d > m_filtered_positions;      
      std::vector<size_t> m_cell_offset;
      Mat3d m_xform;
      uint64_t m_idMin = std::numeric_limits<uint64_t>::max();
      uint64_t m_idMax = 0;
    };

}

