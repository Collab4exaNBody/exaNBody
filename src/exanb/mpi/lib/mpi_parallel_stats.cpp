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
#include <mpi.h>
#include <vector>

#include <exanb/mpi/mpi_parallel_stats.h>

namespace exanb
{

  void mpi_parallel_stats(MPI_Comm comm, const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg)
  {
    minval = x;
    maxval = x;
    avg = x;
    np = 1;
    r = 0;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &np);
    MPI_Allreduce(MPI_IN_PLACE,minval.data(),minval.size(),MPI_DOUBLE,MPI_MIN,comm);
    MPI_Allreduce(MPI_IN_PLACE,maxval.data(),maxval.size(),MPI_DOUBLE,MPI_MAX,comm);
    MPI_Allreduce(MPI_IN_PLACE,avg.data()   ,avg.size()   ,MPI_DOUBLE,MPI_SUM,comm);
    for( double& x : avg ) { x /= np; }
  }

  void mpi_parallel_sum(MPI_Comm comm,
         unsigned long long in,
         unsigned long long &sum,
         unsigned long long &min,
         unsigned long long &avg,
         unsigned long long &max)
  {
    int np = 1;
    MPI_Comm_size(comm, &np);
    MPI_Allreduce(&in,&sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    MPI_Allreduce(&in,&min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);
    MPI_Allreduce(&in,&max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
    avg = sum / np;
  }
    
}
