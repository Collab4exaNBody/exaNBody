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
#include <exanb/mpi/data_types.h>

namespace exanb
{
  

  template<class T, class... U>
  void all_reduce_multi( MPI_Comm comm, MPI_Op op , T , U& ... u)
  {
    T tmp [ sizeof...(U) ];
    int i=0;
    ( ... , ( tmp[i++] = u ) );
    assert( i == sizeof...(U) );
    MPI_Allreduce(MPI_IN_PLACE,tmp,sizeof...(U),mpi_datatype<T>(),op,comm);
    i=0;
    ( ... , ( u = tmp[i++] ) );
    assert( i == sizeof...(U) );
  }

}
