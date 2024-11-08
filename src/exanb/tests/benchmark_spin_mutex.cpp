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
#include <exanb/core/parallel_random.h>
#include <onika/thread.h>
#include <onika/debug.h>

#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>

#include <omp.h>
#include <assert.h>

template<typename spin_lock_t>
void test_spin_lock_impl(const std::string& mesg, size_t factor, ssize_t Ns)
{
  size_t Nb = 1;

# pragma omp parallel
  {
#   pragma omp single
    Nb = omp_get_num_threads();
  }
  Nb *= factor;
  
  std::cout<<"test "<<mesg<<" (Nb="<<Nb<<", Ns="<<Ns<<") ..."<<std::endl; std::cout.flush();
  spin_lock_t* locks = new spin_lock_t[Nb];
  ssize_t* counter = new ssize_t[Nb];
  for(size_t i=0;i<Nb;i++) { counter[i]=0; }
  
  auto T0 = std::chrono::high_resolution_clock::now();
  //auto progressT = T0;

# pragma omp parallel
  {
    auto & re = exanb::rand::random_engine();
    //int ti = omp_get_thread_num();
#   pragma omp barrier

    std::uniform_int_distribution<> idist(0,Nb-1);
#   pragma omp for schedule(static)
    for(ssize_t i=0;i<Ns;i++)
    {
      size_t target = idist(re);
      locks[target].lock();
      counter[target] += 7;
      locks[target].unlock();
    }

#   pragma omp barrier

#   pragma omp for schedule(static)
    for(ssize_t i=0;i<Ns;i++)
    {
      size_t target = idist(re);
      locks[target].lock();
      counter[target] -= 6;
      locks[target].unlock();
    }
  }
  double time_ms = (std::chrono::high_resolution_clock::now()-T0).count()/1000000.0;

  ssize_t sum = 0;
  for(size_t i=0;i<Nb;i++)
  {
    sum += counter[i];
  }
  std::cout<<"sum = "<<sum<<", time = "<<time_ms<<" ms"<<std::endl; std::cout.flush();
  ONIKA_FORCE_ASSERT( sum == Ns );

  delete [] counter;
  delete [] locks;
}

int main(int argc,char*argv[])
{
  
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::string;

  size_t scale = 1;
  size_t Ns = 10000000;
  if( argc < 3 )
  {
    cerr<<"Usage: "<<argv[0]<<" scale samples"<<endl;
    return 1;
  }
  scale = atoi(argv[1]);
  Ns = atoi(argv[2]);

  exanb::rand::generate_seed();

  // first, test spin lock implementations
# ifdef NDEBUG
//  test_spin_lock_impl<null_spin_mutex>("null_spin_mutex",scale,Ns); // _WILL_ fail with asserts enabled. Here for performance comparison only.
# endif  
  test_spin_lock_impl<atomic_lock_spin_mutex<0> >("atomic_lock_spin_mutex<0>",scale,Ns);
  test_spin_lock_impl<atomic_lock_spin_mutex<1000> >("atomic_lock_spin_mutex<1000>",scale,Ns);
  test_spin_lock_impl<stl_spin_mutex>("stl_spin_mutex",scale,Ns);
  test_spin_lock_impl<omp_spin_mutex>("omp_spin_mutex",scale,Ns);
  
  return 0;
}


