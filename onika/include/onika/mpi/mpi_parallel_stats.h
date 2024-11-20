/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <mpi.h>
#include <vector>
#include <cstdlib>

namespace onika
{
  namespace mpi
  {
  
    void mpi_parallel_stats(MPI_Comm comm, const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg);

    void mpi_parallel_sum(MPI_Comm comm,
           unsigned long long in,
           unsigned long long &sum,
           unsigned long long &min,
           unsigned long long &avg,
           unsigned long long &max);

  }
}
