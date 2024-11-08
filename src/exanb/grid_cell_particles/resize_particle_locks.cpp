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
#include <onika/thread.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/math/basic_types.h>
#include <onika/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <memory>

namespace exanb
{

  template<class GridT>
  struct ResizeParticleLocksNode : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT);
    ADD_SLOT( GridParticleLocks , particle_locks , INPUT_OUTPUT );
    ADD_SLOT( spin_mutex_array , flat_particle_locks , INPUT_OUTPUT );

    inline void execute ()  override final
    {      
      IJK dims = grid->dimension();
      size_t n_cells = grid->number_of_cells();
      ldbg << "resize_particle_locks: cells: "<<particle_locks->size() <<" -> " << n_cells << std::endl;
      
      particle_locks->resize( n_cells );
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc)
        {
          particle_locks->at(i).resize( grid->cell_number_of_particles(i) );
        }
        GRID_OMP_FOR_END
      }

      flat_particle_locks->resize( grid->number_of_particles() );
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "resize_particle_locks", make_grid_variant_operator< ResizeParticleLocksNode > );
  }

}

