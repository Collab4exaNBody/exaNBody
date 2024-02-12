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

#include <exanb/mpi/xs_data_move_types.h>

#include <vector>
#include <type_traits>
#include <iterator>
#include <mpi.h>
#include <assert.h>

#ifndef NDEBUG
#include <cstring>
#endif

namespace XsDataMove
{
  
  /*
  input and output may point to the same elements
  */
  
template<typename InputIterator, typename OutputIterator, class = std::enable_if_t< std::is_same<typename std::iterator_traits<InputIterator>::value_type,typename std::iterator_traits<OutputIterator>::value_type>::value > >
static inline void data_move(
    // SYMETRIC IN, common across all processes 
    MPI_Comm comm,
    const std::vector<int>& send_indices,     // resized to localElementCountBefore, contain indices in 'before' array to pack into send buffer
    const std::vector<int>& send_count,       // resized to number of processors in comm, unit data element count (not byte size)
    const std::vector<int>& send_displ,       // resized to number of processors in comm, unit data element count (not byte size)
    const std::vector<int>& recv_indices,     // resized to localElementCountAfter
    const std::vector<int>& recv_count,       // resized to number of processors in comm, unit data element count (not byte size)
    const std::vector<int>& recv_displ,       // resized to number of processors in comm, unit data element count (not byte size)
    
    InputIterator input,                      // input array
    OutputIterator output                     // output array
    )
{
    using DataType = typename std::iterator_traits<InputIterator>::value_type;
    constexpr bool is_same = std::is_same< DataType , typename std::iterator_traits<OutputIterator>::value_type >::value ;
    static_assert( is_same , "Input and output iterator must have the same data type" );
    //assert( is_same ); // TODO: should be compile time error, not execution time error
  
    int rank = -1;
    int nProcs = -1;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&nProcs);

    assert( nProcs >= 1 );
    assert( rank >= 0 && rank < nProcs );
    assert( send_count.size() >= static_cast<size_t>(nProcs) );
    assert( send_displ.size() >= static_cast<size_t>(nProcs) );
    assert( recv_count.size() >= static_cast<size_t>(nProcs) );
    assert( recv_displ.size() >= static_cast<size_t>(nProcs) );
    assert( send_indices.size() == static_cast<size_t>(send_count[nProcs-1] + send_displ[nProcs-1]) );
    assert( recv_indices.size() == static_cast<size_t>(recv_count[nProcs-1] + recv_displ[nProcs-1]) );

    int sendSize = send_indices.size();
    int recvSize = recv_indices.size();
    std::vector<DataType> comm_input( sendSize ); // temporary send buffer
    std::vector<DataType> comm_output( recvSize ); // temporary receive buffer
    
    // re-arrange data prior to AllToAll directive (data must be grouped by target processor)
    for(int i=0 ; i<sendSize ; i++ , ++input )
    {
#     ifndef NDEBUG
      int src_index = send_indices[i];
      assert( src_index>=0 && src_index<sendSize );
#     endif
      comm_input[ send_indices[i] ] =  *input;
    }

#   ifndef NDEBUG
    std::memset( reinterpret_cast<void*>(comm_output.data()), 0xFF, sizeof(DataType)*recvSize );
#   endif

    switch( sizeof(DataType) )
    {
      case sizeof(char):
        MPI_Alltoallv( (char*) comm_input.data() , (int*) send_count.data(), (int*) send_displ.data() , MPI_CHAR , (char*) comm_output.data() , (int*) recv_count.data() , (int*) recv_displ.data() , MPI_CHAR , comm );
        break;
        
      case sizeof(float):
        MPI_Alltoallv( (float*) comm_input.data() , (int*) send_count.data(), (int*) send_displ.data() , MPI_FLOAT , (float*) comm_output.data() , (int*) recv_count.data() , (int*) recv_displ.data() , MPI_FLOAT , comm );
        break;
        
      case sizeof(double):
        MPI_Alltoallv( (double*) comm_input.data() , (int*) send_count.data(), (int*) send_displ.data() , MPI_DOUBLE , (double*) comm_output.data() , (int*) recv_count.data() , (int*) recv_displ.data() , MPI_DOUBLE , comm );
        break;

      default:
        {
            MPI_Datatype element_type; 
            MPI_Type_contiguous(sizeof(DataType), MPI_CHAR, &element_type);
            MPI_Type_commit(&element_type);
            MPI_Alltoallv( comm_input.data() , (int*) send_count.data(), (int*) send_displ.data() , element_type , comm_output.data() , (int*) recv_count.data() , (int*) recv_displ.data() , element_type , comm );
            MPI_Type_free(&element_type);
        }
        break;
    }

    for(int i=0 ; i<recvSize ; i++ , ++output )
    {
#     ifndef NDEBUG
      int dst_index = recv_indices[i];
      assert( dst_index>=0 && dst_index<recvSize );
#     endif
      *output = comm_output[ recv_indices[i] ];
    }
}
      
} // namespace XsDataMove
