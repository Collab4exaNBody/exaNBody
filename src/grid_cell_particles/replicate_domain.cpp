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
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/geometry.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/log.h>
#include <onika/thread.h>
#include <exanb/grid_cell_particles/replicate_domain.h>

#include <vector>
#include <iostream>
#include <mpi.h>

namespace exanb
{
  template<class GridT> using ReplicateDomainDefaultIdShift = ReplicateDomain<GridT>;

   // === register factories ===
  ONIKA_AUTORUN_INIT(replicate_domain)
  {
    OperatorNodeFactory::instance()->register_factory( "replicate_domain", make_grid_variant_operator< ReplicateDomainDefaultIdShift > );
  }

}

