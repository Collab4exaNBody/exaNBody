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

#include <memory>
#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/parallel_build_dual_neighbors.h>

namespace exanb
{
  

  struct ComputeDualNeighborsNode : public OperatorNode
  {
    ADD_SLOT(GridParticleNeighbors , primary_neighbors , INPUT );
    ADD_SLOT(GridParticleNeighbors , dual_neighbors    , INPUT_OUTPUT );

    inline void execute () override final
    {
      GridParticleNeighbors& pb = *primary_neighbors;
      GridParticleNeighbors& db = *dual_neighbors;
      parallel_build_dual_neighbors( pb, db );

#     ifndef NDEBUG
      assert( check_dual_neighbors(db) );
      size_t n_cells = db.size();
      assert( pb.size() == n_cells );
      size_t n_pb = 0;
      size_t n_db = 0;
      for(size_t i=0;i<n_cells;i++)
      {
        n_pb += pb[i].neighbors.size();
        n_db += db[i].neighbors.size();
      }
      assert( n_pb == n_db );
      ldbg<<"dual neighbors count = "<<n_db<<std::endl;
#     endif
    }
  };


  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "compute_dual_neighbors",
      make_compatible_operator< ComputeDualNeighborsNode >
      );
  }
  
}

