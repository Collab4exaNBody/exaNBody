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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/grid.h>
#include <exanb/fields.h>

#include <mpi.h>
#include <vector>
#include <string>

namespace exanb
{
  

  using std::vector;
  using std::string;
  using std::endl;

  struct CommBTreeNode : public OperatorNode
  {
    ADD_SLOT( MPI_Comm              , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( std::vector<MPI_Comm> , mpi_btree , OUTPUT);
    
    /*
      Important Notes :
      this algorithm recursively splits input communicator.
      it guarantees that the left half has less or equal participants than right part.
      it guarantees that the right part is either equal size than left part or just +1.
      l <= r
      r == l or r == l+1
    */
    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      std::vector<MPI_Comm>& comm_btree = *mpi_btree;
      comm_btree.clear();
      comm_btree.reserve(1024);

      int size = 1;      
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&size);
      string comm_name = "W";

      while( size >= 2 )
      {
        comm_btree.push_back(comm);
        
        MPI_Comm newcomm = MPI_COMM_NULL;
        int newsize = 0;
        int newrank = 0;
        string newname = comm_name;

        bool side = ( rank >= (size/2) );
        if( side )
        {
          newrank = rank - (size/2);
          newsize = size - (size/2);
          newname += "_B";
        }
        else
        {
          newrank = rank;
          newsize = size/2;
          newname += "_A";
        }

        MPI_Comm_split( comm , side , rank , &newcomm);
        MPI_Comm_rank(newcomm,&newrank);
        MPI_Comm_size(newcomm,&newsize);
        ldbg<< comm_name << " (size=" << size << ", rank=" << rank <<", side="<<side<< ") -> "<< newname << " (size=" <<newsize<<", rank="<<newrank<<")"<< endl;

        comm_name = newname;
        size = newsize;
        rank = newrank;
        comm = newcomm;
      }
    }
  };


  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "mpi_btree", make_compatible_operator<CommBTreeNode> );
  }

}

