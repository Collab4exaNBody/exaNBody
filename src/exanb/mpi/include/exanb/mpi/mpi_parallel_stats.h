#pragma once

#include <mpi.h>
#include <vector>
#include <cstdlib>

namespace exanb
{

  void mpi_parallel_stats(MPI_Comm comm, const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg);

  void mpi_parallel_sum(MPI_Comm comm,
         unsigned long long in,
         unsigned long long &sum,
         unsigned long long &min,
         unsigned long long &avg,
         unsigned long long &max);
}
