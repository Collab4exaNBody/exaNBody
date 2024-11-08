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
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct AmrGridSubCellPairs : public OperatorNode
  {
    ADD_SLOT( AmrGrid , amr            , INPUT , REQUIRED );
    ADD_SLOT( double  , nbh_dist       , INPUT , REQUIRED );  // value added to the search distance to update neighbor list less frequently
    ADD_SLOT( Domain  , domain         , INPUT_OUTPUT );
    ADD_SLOT( AmrSubCellPairCache , amr_grid_pairs , INPUT_OUTPUT );

    inline void execute () override final
    {
      max_distance_sub_cell_pairs( ldbg , *amr , domain->cell_size() , *nbh_dist , *amr_grid_pairs );
    }

  private:  
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("amr_grid_pairs", make_simple_operator< AmrGridSubCellPairs > );
  }

}

