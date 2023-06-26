#pragma once

#include <exanb/core/basic_types.h>
#include <vector>

namespace exanb
{

  struct CpuGpuPartition
  {
    GridBlock m_cpu_block;
    GridBlock m_gpu_block;
    double m_gpu_cost_threshold = 0;
    size_t cpu_cell_count = 0;
    size_t gpu_cell_particles_threshold = 0;
  };

}

