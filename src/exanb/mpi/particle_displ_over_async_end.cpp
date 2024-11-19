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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/math/basic_types.h>

#include <cmath>
#include <mpi.h>

#include <exanb/mpi/particle_displ_over_async_request.h>

namespace exanb
{
  

  struct ParticleDisplOverAsyncEnd : public OperatorNode
  {
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( ParticleDisplOverAsyncRequest , particle_displ_comm , INPUT_OUTPUT );
    ADD_SLOT( bool               , result    , OUTPUT );

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
compute the distance between each particle in grid input and it's backup position in backup_r input.
sets result output to true if at least one particle has moved further than threshold.
)EOF";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
//      MPI_Status status;
//      MPI_Wait( &(particle_displ_comm->m_request), &status );
      particle_displ_comm->wait();
      ldbg << "particles over threshold ="<< particle_displ_comm->m_particles_over <<" / "<< particle_displ_comm->m_all_particles_over << std::endl;
      *result = ( particle_displ_comm->m_all_particles_over > 0 );
    }
  };
    

  // === register factories ===  
  ONIKA_AUTORUN_INIT(particle_displ_over_async_end)
  {
   OperatorNodeFactory::instance()->register_factory( "particle_displ_over_async_end", make_simple_operator< ParticleDisplOverAsyncEnd > );
  }

}

