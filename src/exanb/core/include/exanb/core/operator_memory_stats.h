#pragma once

#include <exanb/core/operator.h>
#include <mpi.h>

namespace exanb
{
  void print_operator_memory_stats( exanb::OperatorNode* simulation_graph , MPI_Comm comm , size_t musage_threshold = 1024 );
}
