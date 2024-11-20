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
#include <cassert>

namespace onika
{
  namespace mpi
  {

    template<typename T> inline MPI_Datatype mpi_datatype() { std::abort(); return MPI_CHAR;}
    template<> inline MPI_Datatype mpi_datatype<char>() { return MPI_CHAR; }
    template<> inline MPI_Datatype mpi_datatype<unsigned char>() { return MPI_UNSIGNED_CHAR; }
    template<> inline MPI_Datatype mpi_datatype<short>() { return MPI_SHORT; }
    template<> inline MPI_Datatype mpi_datatype<unsigned short>() { return MPI_UNSIGNED_SHORT; }
    template<> inline MPI_Datatype mpi_datatype<int>() { return MPI_INT; }
    template<> inline MPI_Datatype mpi_datatype<unsigned int>() { return MPI_UNSIGNED; }
    template<> inline MPI_Datatype mpi_datatype<long>() { return MPI_LONG; }
    template<> inline MPI_Datatype mpi_datatype<unsigned long>() { return MPI_UNSIGNED_LONG; }
    template<> inline MPI_Datatype mpi_datatype<float>() { return MPI_FLOAT; }
    template<> inline MPI_Datatype mpi_datatype<double>() { return MPI_DOUBLE; }
    
    template<class T>
    static inline MPI_Datatype mpi_typeof(const T&) { return mpi_datatype<T>(); }

  } 
}

