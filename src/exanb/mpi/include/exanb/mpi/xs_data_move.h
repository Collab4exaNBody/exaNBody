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
#include <mpi.h>

namespace XsDataMove
{

  void communication_scheme_from_ids(
      // SYMETRIC IN, common across all processes 
      MPI_Comm comm,
      id_type allIdMin, // included
      id_type allIdMax, // excluded

      // IN, differs across processes
      size_type localElementCountBefore,
      const id_type* localIdsBefore,
      size_type localElementCountAfter,
      const id_type* localIdsAfter,

      // OUT
      std::vector<int>& send_indices,     // resized to localElementCountBefore, contain indices in 'before' array to pack into send buffer
      std::vector<int>& send_count,       // resized to number of processors in comm, unit data element count (not byte size)
      std::vector<int>& send_displ,       // resized to number of processors in comm, unit data element count (not byte size)
      std::vector<int>& recv_indices,     // resized to localElementCountAfter
      std::vector<int>& recv_count,       // resized to number of processors in comm, unit data element count (not byte size)
      std::vector<int>& recv_displ        // resized to number of processors in comm, unit data element count (not byte size)
      );

} // namespace XsDataMove

#include <exanb/mpi/xs_data_move_impl.h>

