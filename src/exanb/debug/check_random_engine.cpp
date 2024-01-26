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
#include <exanb/core/parallel_random.h>

#include <mpi.h>

namespace exanb
{
  
  
  struct CheckRandomEngine : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi     , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( long     , cycles  , INPUT , 10 );
    ADD_SLOT( long     , samples , INPUT , 20 );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&nprocs);
      
      int nsamples = *samples;
      int ncycles = *cycles;
      
      for(int c=0;c<ncycles;c++)
      {
        int tab[nsamples];
        for(int i=0;i<nsamples;i++)
        {
          tab[i] = -1;
        }
#       pragma omp parallel
        {
          auto& re = rand::random_engine();
          std::uniform_int_distribution<int> rint(1000,9999);

#         pragma omp for schedule(static)
          for(int i=0;i<nsamples;i++)
          {
            tab[i] = rint(re);
          }
        }
        
        lout << "cycle "<<c<<" :";
        for(int i=0;i<nsamples;i++)
        {
          lout <<' ' << tab[i];
        }
        lout << std::endl;
      }
    }

  };
  
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "check_random_engine", make_simple_operator< CheckRandomEngine > );
  }

}

