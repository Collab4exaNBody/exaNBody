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
#include <cstring>

namespace exanb
{
  template<typename T>
  bool all_value_equal(MPI_Comm comm, const T& value)
  {
    T tmp = value;
    MPI_Bcast((char*)&tmp,sizeof(T),MPI_CHAR,0,comm);
    int result = std::memcmp( &value, &tmp, sizeof(T) ) == 0;
    MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MIN,comm);
    return result;
  }

  template<typename T>
  bool all_value_equal(MPI_Comm comm, const std::vector<T>& value)
  {
    std::vector<T> tmp = value;
    MPI_Bcast((char*)tmp.data(),sizeof(T)*tmp.size(),MPI_CHAR,0,comm);
    int result = std::memcmp( value.data(), tmp.data(), sizeof(T)*tmp.size() ) == 0;
    MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MIN,comm);
    return result;
  }
    
}
