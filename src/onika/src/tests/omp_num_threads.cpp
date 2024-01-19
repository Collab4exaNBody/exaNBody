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

int main(int argc,char*argv[])
{

# pragma omp parallel
  {
    int nt = omp_get_num_threads();
    _Pragma("omp single") std::cout<<"nt="<<nt<<"\n";
    int tid = omp_get_thread_num();
    _Pragma("omp critical(dbg_mesg)") std::cout<<"thread "<<tid<<" / "<<nt<<"\n";
  }

  return 0;
}

