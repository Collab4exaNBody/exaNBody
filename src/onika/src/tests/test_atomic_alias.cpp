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
#include <atomic>
#include <vector>
#include <cassert>

#include <onika/cuda/cuda_math.h>

template<class T>
void test_atomic_alias( T )
{
  static_assert( sizeof(T) == sizeof(std::atomic<T>) && alignof(T) == alignof(std::atomic<T>) );
  constexpr int N = 10;
  std::vector< std::atomic<T> > va(N);
  std::vector< T > v(N);
  for(int i=0;i<N;i++) { va[i]=0; v[i]=0; }
  
  auto atomic_inc_alias = [&](int i)->void { reinterpret_cast< std::atomic<T>* >( & v[i] )->fetch_add( 1 , std::memory_order_relaxed); };

# pragma omp parallel
  {
    for(int i=0;i<99999;i++)
    {
      va[i%N].fetch_add( 1 , std::memory_order_relaxed );
      atomic_inc_alias( i%N );
    }
  }

  for(int i=0;i<N;i++)
  {
    std::cout<<i<<" : va="<<va[i]<<" , v="<<v[i]<<std::endl;
    assert( va[i] == v[i] );
  }
}

int main()
{
  test_atomic_alias( int{} );
  test_atomic_alias( size_t{} );

  std::cout << "size_t max = "<<std::numeric_limits<size_t>::max()<<" / "<< onika::cuda::numeric_limits<size_t>::max << std::endl;

  return 0;
}

