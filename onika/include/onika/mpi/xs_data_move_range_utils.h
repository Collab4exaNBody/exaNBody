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

#include <onika/mpi/xs_data_move_types.h>
#include <cassert>

namespace onika
{
  namespace mpi
  {

    /*!
    @param id Id to get target processor for, based on the Id range assigned to this processor
    @param nProcs Number of participating processes
    @param idMin Minimum id value across all processes
    @param idMax Maximum id value across all processes
    @return the process in charge for the range containing specified Id value
    */
    static inline int get_process_for_id( id_type id, int nProcs, id_type idMin, id_type idMax )
    {
      assert( idMax > idMin );
      assert( id >= idMin && id < idMax );
      assert( nProcs >= 1 );
      return ( (id-idMin+1) * nProcs - 1) / ( idMax - idMin );
    }


    static inline id_type get_id_range_start_for_process( int rank, int nProcs, id_type idMin, id_type idMax )
    {
      assert( idMax > idMin );
      assert( nProcs >= 1 );
      assert( rank >= 0 && rank < nProcs );
      size_type range_size = idMax - idMin;
      return idMin + ( (rank*range_size) / nProcs );
    }

    static inline id_type get_id_range_end_for_process( int rank, int nProcs, id_type idMin, id_type idMax )
    {
      assert( idMax > idMin );
      assert( nProcs >= 1 );
      assert( rank >= 0 && rank < nProcs );
      size_type range_size = idMax - idMin;
      return idMin + ( ((rank+1)*range_size) / nProcs );
    }

    static inline size_type get_id_range_size_for_process( int rank, int nProcs, id_type idMin, id_type idMax )
    {
      assert( idMax > idMin );
      assert( nProcs >= 1 );
      assert( rank >= 0 && rank < nProcs );
      id_type range_start = get_id_range_start_for_process( rank, nProcs, idMin, idMax );
      id_type range_end = get_id_range_end_for_process( rank, nProcs, idMin, idMax );
      return range_end - range_start;
    }

    static inline bool id_in_process_id_range( id_type id, id_type idMin, id_type idMax, int rank, int nProcs )
    {
      assert( idMax > idMin );
      assert( id >= idMin && id < idMax );
      assert( nProcs >= 1 );
      assert( rank >= 0 && rank < nProcs );
      id_type range_start = get_id_range_start_for_process( rank, nProcs, idMin, idMax );
      id_type range_end = get_id_range_end_for_process( rank, nProcs, idMin, idMax );
      return id>=range_start && id<range_end;
    }

  } // namespace mpi
} // namespace onika



// ================================================
// =================== UNIT TESTS =================
// ================================================

#include <onika/test/unit_test.h>

ONIKA_UNIT_TEST(xsdatamove_process_id_range)
{
  onika::mpi::id_type allIdMin = 5;
  onika::mpi::id_type allIdMax = 1000;
  for( int nProcs = 1 ; nProcs < 16 ; nProcs++ )
  {
    for( onika::mpi::id_type i=allIdMin ; i<allIdMax ; i++ )
    {
      int process_for_id = onika::mpi::get_process_for_id( i, nProcs, allIdMin, allIdMax );
      ONIKA_TEST_ASSERT( onika::mpi::id_in_process_id_range( i, allIdMin, allIdMax, process_for_id, nProcs) );
    }
  }
}

ONIKA_UNIT_TEST(xsdatamove_process_id_range_size)
{
  for( onika::mpi::id_type allIdMin = 0 ; allIdMin < 2 ; allIdMin++ )
  {
    for( onika::mpi::id_type allIdMax = allIdMin+1 ; allIdMax < 1002 ; allIdMax+=103 )
    {
      ONIKA_TEST_ASSERT( allIdMax > allIdMin );
      for( int nProcs = 1 ; nProcs < 16 ; nProcs++ )
      {
        onika::mpi::size_type count = 0;
        for(int i=0;i<nProcs;i++)
        {
          count += onika::mpi::get_id_range_size_for_process(i,nProcs,allIdMin,allIdMax);
        }
        ONIKA_TEST_ASSERT( count == (allIdMax-allIdMin) );
      }
    }
  }
}

