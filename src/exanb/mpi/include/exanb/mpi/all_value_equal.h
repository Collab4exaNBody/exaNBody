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
