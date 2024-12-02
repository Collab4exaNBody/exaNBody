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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/fields.h>
#include <exanb/core/domain.h>

#include <exanb/core/cell_costs.h>
#include <exanb/mpi/cost_weight_map.h>

#include <vector>
#include <algorithm>
#include <limits>

namespace exanb
{

  // simple cost model where the cost of a cell is the number of particles in it
  // 
  template< class GridT
          , class WeightingFieldT = int // no field
          >
  class SimpleCostModel : public OperatorNode
  {
    using has_weighting_field_t = typename GridT:: template HasField <WeightingFieldT>;
    static inline constexpr bool has_weighting_field = has_weighting_field_t::value;
    using DoubleVector = std::vector<double>;

    ADD_SLOT( GridT , grid ,INPUT);
    ADD_SLOT( DoubleVector , cost_model_coefs , INPUT_OUTPUT, DoubleVector({0.0,0.0,1.0,0.0}) , DocString{"Polynomial coefs for cost function coefficients. coefficients for resepctively X³ ,X², X and Const. X is particle density"} ); 
    ADD_SLOT( CostWeightMap , cost_weight_map , INPUT, OPTIONAL );
    
    ADD_SLOT( CellCosts , cell_costs ,OUTPUT );

  public:

    inline void execute () override final
    {
      GridT& grid = *(this->grid);
      CellCosts& cell_costs = *(this->cell_costs);

      cost_model_coefs->resize(4,0.0);      
      const double d3 = cost_model_coefs->at(0);  // X^3
      const double d2 = cost_model_coefs->at(1);  // X^2
      const double d1 = cost_model_coefs->at(2);  // X^1
      const double cc = cost_model_coefs->at(3);  // X^0

      const int ghost_layers = grid.ghost_layers();
      const double cell_size = grid.cell_size();
      const double cell_volume = cell_size * cell_size * cell_size;

      ldbg << "SimpleCostModel: cc="<< cc << ", d1="<<d1<<", d2="<<d2<<", d3="<<d3<<", vol="<<cell_volume<< std::endl;
      
      std::vector<double> cost_weights;
      const double * __restrict__ cweight = nullptr;
      
      if constexpr (has_weighting_field)
      {
        if( cost_weight_map.has_value() )
        {
          unsigned long max_id = 0;
          for(const auto& it:*cost_weight_map)
          {
            ldbg << "WEIGHT MAP: "<<it.first << " -> "<<it.second<<std::endl;
            if(it.first >= max_id ) max_id = it.first+1;
          }
          if( max_id>0 )
          {
            cost_weights.assign( max_id , 1.0 );
            for(const auto& it:*cost_weight_map)
            {
              cost_weights[it.first] = it.second;
            }
            cweight = cost_weights.data();
          }
        }
      }

      // Warning: for correctness we only account for inner cells (not ghost cells)
      IJK grid_dims = grid.dimension();
      auto cells = grid.cells();
      cell_costs.m_block.start = grid.offset() + ghost_layers;
      cell_costs.m_block.end = ( grid.offset() + grid_dims ) - ghost_layers;

      assert( cell_costs.m_block.end.i >= cell_costs.m_block.start.i );
      assert( cell_costs.m_block.end.j >= cell_costs.m_block.start.j );
      assert( cell_costs.m_block.end.k >= cell_costs.m_block.start.k );

      IJK dims = dimension( cell_costs.m_block );
      assert( dims == (grid_dims-2*ghost_layers) );
      const size_t n_cells = grid_cell_count( dims );
      cell_costs.m_costs.resize( n_cells , 0. );

      double min_val = std::numeric_limits<double>::max();
      double max_val = 0.0;
      size_t max_np = 0;
      size_t min_np = grid.number_of_particles();

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(static) reduction(min:min_val) reduction(max:max_val) reduction(max:max_np) reduction(min:min_np) )
        {
          const size_t cell_i = grid_ijk_to_index( grid_dims , loc + ghost_layers );
          assert( cell_i >= 0 && cell_i < grid.number_of_cells() );
          assert( i >= 0 && static_cast<size_t>(i) < cell_costs.m_costs.size() );
          const size_t N = cells[cell_i].size();
          double np = N;
          if constexpr ( has_weighting_field )
          {
            static constexpr onika::soatl::FieldId<WeightingFieldT> WeightingField = {};
            if ( cweight != nullptr )
            {
              np = 0.0;
              for(size_t j=0;j<N;j++)
              {
                np += cweight[ static_cast<unsigned long>( cells[cell_i][WeightingField][j] ) ];
              }
            }
          }

          const double pvol = np / cell_volume;
          const double cost = pvol*d1 + pvol*pvol*d2 + pvol*pvol*pvol*d3 + cc;
          cell_costs.m_costs[i] = cost;

          max_np = std::max( max_np , N );
          min_np = std::min( min_np , N );
          min_val = std::min( min_val , cost );
          max_val = std::max( max_val , cost );
        }
        GRID_OMP_FOR_END
      }

      ldbg << "SimpleCostModel: min="<<min_val<<", max="<<max_val<<", p/c=["<<min_np<<','<<max_np<<"]" << std::endl;
    }
        
  };

}

