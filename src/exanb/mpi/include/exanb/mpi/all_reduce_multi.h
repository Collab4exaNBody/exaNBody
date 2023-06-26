#pragma once

#include <mpi.h>
#include <exanb/core/value_streamer.h>

#include <exanb/mpi/data_types.h>

namespace exanb
{
  

  template<class T, class... U>
  void all_reduce_multi( MPI_Comm comm, MPI_Op op , T , U& ... u)
  {
    T tmp [ sizeof...(U) ];
    ( ValueStreamer<T>( tmp ) << ... << u );
    MPI_Allreduce(MPI_IN_PLACE,tmp,sizeof...(U),mpi_datatype<T>(),op,comm);
    ( ValueStreamer<T>( tmp ) >> ... >> u );
  }

}
