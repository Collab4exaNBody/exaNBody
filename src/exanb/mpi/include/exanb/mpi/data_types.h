#pragma once

#include <mpi.h>
#include <vector>
#include <cassert>

namespace exanb
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

