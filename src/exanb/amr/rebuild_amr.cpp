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
#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/amr/amr_grid.h>
#include <onika/math/basic_types.h>
#include <exanb/amr/amr_grid_algorithm.h>

#include <memory>



namespace exanb
{

  template<typename GridT>
  struct RebuildAmrNode : public OperatorNode
  {
    ADD_SLOT( double  , sub_grid_density , INPUT , 5.0 );
    ADD_SLOT( long    , enforced_ordering, INPUT, 1 );
    ADD_SLOT( GridT   , grid             , INPUT_OUTPUT );
    ADD_SLOT( AmrGrid , amr              , INPUT_OUTPUT );

    inline void execute () override final
    { 
      static constexpr std::integral_constant<unsigned int,1> subres1 = {};
      static constexpr std::integral_constant<unsigned int,2> subres2 = {};
      static constexpr std::integral_constant<unsigned int,3> subres3 = {};
      static constexpr std::integral_constant<unsigned int,4> subres4 = {};
      static constexpr std::false_type no_z_order = {};

      ldbg << "RebuildAmr: sub_grid_density="<< *sub_grid_density << " , ncells="<<grid->number_of_cells() <<std::endl;

      amr->clear_sub_grids( grid->number_of_cells() );      
      amr->m_z_curve = false;
      switch( *enforced_ordering )
      {
        case 2 : rebuild_sub_grids( ldbg, *grid, *amr, *sub_grid_density, no_z_order, subres2 ); break;
        case 3 : rebuild_sub_grids( ldbg, *grid, *amr, *sub_grid_density, no_z_order, subres3 ); break;
        case 4 : rebuild_sub_grids( ldbg, *grid, *amr, *sub_grid_density, no_z_order, subres4 ); break;
        default: rebuild_sub_grids( ldbg, *grid, *amr, *sub_grid_density, no_z_order, subres1 ); break;
      }
    }

  };

 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "rebuild_amr", make_grid_variant_operator< RebuildAmrNode > );
  }
}

