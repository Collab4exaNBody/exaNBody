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
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include <thread>
#include <set>

#define TEST_ASSERT(cond) if(!(cond)) std::abort()

std::set<std::thread::id> involved_threads;

void report_task_threads(int task)
{
      int tid = omp_get_thread_num();
#     pragma omp critical
      {
        involved_threads.insert( std::this_thread::get_id() );
        std::cout<<"thread #"<<tid<<" (id=" << std::this_thread::get_id() <<") in task #"<<task<<std::endl;
      }    
}

int main(int argc,char*argv[])
{
  unsigned int nteams = 2;
  if(argc>=2)
  {
    nteams = std::atoi(argv[1]);
  }
  TEST_ASSERT( nteams>=1 );

  std::cout << "-------- omp teams -------" << std::endl;

  int n_threads = 0;
  int threads_idx_sum = 0;
# pragma omp parallel
  {
    int tid = omp_get_thread_num();
#   pragma omp critical
    {
      int nt = omp_get_num_threads();
      TEST_ASSERT( n_threads==0 || n_threads==nt );
      n_threads = nt;
    }
#   pragma omp atomic
    threads_idx_sum += tid;
  }
  TEST_ASSERT( threads_idx_sum == (n_threads*(n_threads-1))/2 );	

  int task_threads = n_threads / nteams;
  TEST_ASSERT( task_threads>=1 );

  std::cout << "nteams="   <<nteams<<", n_threads="<<n_threads<<", task_threads="<<task_threads<<std::endl;

  int res = 0;
  int n = 0;
# pragma omp target teams num_teams(nteams) map(res, n) reduction(+:res)
  {
    res = omp_get_team_num();
    if (omp_get_team_num() == 0)
    {
      n = omp_get_num_teams();
    }
  }
  std::cout << "n="<<n<<", res="<<res<<std::endl;

  return 0;
}

