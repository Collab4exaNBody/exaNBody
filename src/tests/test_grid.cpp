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
// this is a test, it always needs enabled asserts
#ifndef NDEBUG
#define NDEBUG 1
#endif

#include <exanb/core/grid.h>
#include <exanb/fields.h>

#include <iostream>
#include <cstdlib>

struct A
{
  int memory_bytes(int y);
};

int main(int argc,char*argv[])
{
  using std::cout;
  using std::endl;
  

  auto grid = make_grid( field::ep, field::ax, field::ay, field::az, field::vx, field::vy, field::vz, field::id, field::type );
  std::cout<<"grid.memory_bytes() = "<<grid.memory_bytes()<<std::endl;

  static_assert( onika::memory::has_memory_bytes_method_v<decltype(grid)> , "a valid memory_bytes method is needed for proper memory usage accounting" );
  A a;
  static_assert( ! onika::memory::has_memory_bytes_method_v<decltype(a)> , "shouldn't be detected as having a valid memorty_bytes method");

  return 0;
}


