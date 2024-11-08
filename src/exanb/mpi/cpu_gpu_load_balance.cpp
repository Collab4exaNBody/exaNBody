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
#include <onika/log.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>

#include <exanb/core/cell_costs.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/cpu_gpu_partition.h>

#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#endif

#include <vector>
#include <string>
#include <algorithm>

namespace exanb
{
  

  struct CpuGpuLoadBalance : public OperatorNode
  {
    ADD_SLOT( CellCosts , cell_costs      , INPUT , REQUIRED );
    ADD_SLOT( double    , cpu_gpu_perf_ratio, INPUT , 8.0 );
    ADD_SLOT( long      , gpu_cell_particles_threshold , INPUT , 16 );

    ADD_SLOT( CpuGpuPartition , cpu_gpu_partition  , INPUT_OUTPUT);

    inline void execute () override final
    {    
      const CellCosts& grid_costs = *cell_costs;       
      const IJK grid_dims = dimension(grid_costs.m_block);
      size_t n_cells = grid_costs.m_costs.size();
      IJK block_start = {0,0,0};
      IJK block_end = grid_dims;

      double min_cost = std::numeric_limits<double>::max();
      double max_cost = 0.0;
      GRID_FOR_BEGIN(grid_dims,cell_i,cell_loc)
      {
        min_cost = std::min( min_cost , grid_costs.m_costs[cell_i] );
        max_cost = std::max( max_cost , grid_costs.m_costs[cell_i] );
      }
      GRID_FOR_END
   
#     ifdef ONIKA_CUDA_VERSION
      if( parallel_execution_context()->has_gpu_context() )
      {
        //const size_t n_cells = grid_cell_count(grid_dims);

        ldbg << "---- CpuGpuLoadBalance: grid_dims=" << grid_dims  << " ----" << std::endl;
        
        std::vector<double> i_cost( grid_dims.i , 0. );
        std::vector<double> j_cost( grid_dims.j , 0. );
        std::vector<double> k_cost( grid_dims.k , 0. );
                     
        GRID_FOR_BEGIN(grid_dims,_,cell_loc)
        {
          size_t cell_i = grid_ijk_to_index( grid_dims, cell_loc + block_start );
          assert( cell_loc.i>=0 && cell_loc.i < static_cast<ssize_t>(i_cost.size()) );
          assert( cell_loc.j>=0 && cell_loc.j < static_cast<ssize_t>(j_cost.size()) );
          assert( cell_loc.k>=0 && cell_loc.k < static_cast<ssize_t>(k_cost.size()) );
          i_cost[ cell_loc.i ] += grid_costs.m_costs[cell_i];
          j_cost[ cell_loc.j ] += grid_costs.m_costs[cell_i];
          k_cost[ cell_loc.k ] += grid_costs.m_costs[cell_i];
        }
        GRID_FOR_END

        BoxSplit bs;
        
        bs.i_split = find_best_split( i_cost, 1.0, *cpu_gpu_perf_ratio );
        assert( bs.i_split.position >= 0 && static_cast<ssize_t>(bs.i_split.position) < grid_dims.i );

        bs.j_split = find_best_split( j_cost, 1.0, *cpu_gpu_perf_ratio );
        assert( bs.j_split.position >= 0 && static_cast<ssize_t>(bs.j_split.position) < grid_dims.j );

        bs.k_split = find_best_split( k_cost, 1.0, *cpu_gpu_perf_ratio );
        assert( bs.k_split.position >= 0 && static_cast<ssize_t>(bs.k_split.position) < grid_dims.k );
      
        GridBlock left_block = { block_start , block_end };
        GridBlock right_block = { block_start , block_end };

        bool i_split_valid = ( grid_dims.i >= 2 ) && ( bs.i_split.position>0 ) && ( static_cast<ssize_t>(bs.i_split.position) < grid_dims.i );
        bool j_split_valid = ( grid_dims.j >= 2 ) && ( bs.j_split.position>0 ) && ( static_cast<ssize_t>(bs.j_split.position) < grid_dims.j );
        bool k_split_valid = ( grid_dims.k >= 2 ) && ( bs.k_split.position>0 ) && ( static_cast<ssize_t>(bs.k_split.position) < grid_dims.k );

        if( bs.i_split.worst_balance < bs.j_split.worst_balance && bs.i_split.worst_balance < bs.k_split.worst_balance && i_split_valid )
        {
          //ldbg<<"--- level "<<level<<" split I at "<<bs.i_split.position<<" ---" << std::endl;
          left_block.end.i = block_start.i + bs.i_split.position;
          right_block.start.i = block_start.i + bs.i_split.position;
        }
        else if( bs.j_split.worst_balance < bs.i_split.worst_balance && bs.j_split.worst_balance < bs.k_split.worst_balance && j_split_valid )
        {
          //ldbg<<"--- level "<<level<<" split J at "<<bs.j_split.position<<" ---" << std::endl;
          left_block.end.j = block_start.j + bs.j_split.position;
          right_block.start.j = block_start.j + bs.j_split.position;
        }
        else if( k_split_valid )
        {
          //ldbg<<"--- level "<<level<<" split K at "<<bs.k_split.position<<" ---" << std::endl;
          left_block.end.k = block_start.k + bs.k_split.position;
          right_block.start.k = block_start.k + bs.k_split.position;
        }
        else
        {
          ldbg << "impossible partitioning, splitting regardless of costs : dims="<<grid_dims<< std::endl;
          if( grid_dims.i >= grid_dims.j && grid_dims.i >= grid_dims.k )
          {
            //ldbg << "split I"  << std::endl;
            assert( grid_dims.i >= 2 );
            left_block.end.i = block_start.i + grid_dims.i/2;
            right_block.start.i = left_block.end.i;
            
          }
          else if( grid_dims.j >= grid_dims.i && grid_dims.j >= grid_dims.k )
          {
            //ldbg << "split J"  << std::endl;
            assert( grid_dims.j >= 2 );
            left_block.end.j = block_start.j + grid_dims.j/2;
            right_block.start.j = left_block.end.j;
          }
          else
          {
            //ldbg << "split K"  << std::endl;
            assert( grid_dims.k >= 2 );
            left_block.end.k = block_start.k + grid_dims.k/2;
            right_block.start.k = left_block.end.k;
          }
        }
        assert( ! is_empty(left_block) );
        assert( ! is_empty(right_block) );

        cpu_gpu_partition->m_cpu_block = left_block;
        cpu_gpu_partition->m_gpu_block = right_block;

        // compute cost threshold point
        static constexpr size_t hsize = 1024;
        std::vector<double> cell_cost_hist( hsize , 0.0 );
        GRID_FOR_BEGIN(grid_dims,_,cell_loc)
        {
          size_t cell_i = grid_ijk_to_index( grid_dims, cell_loc + block_start );
          ssize_t b = static_cast<ssize_t>( ( grid_costs.m_costs[cell_i] - min_cost ) * hsize / (max_cost-min_cost) ) ;
          if( b < 0 ) { b = 0; }
	  if( b >= ssize_t(hsize) ) { b = hsize-1; }
          cell_cost_hist[b] += grid_costs.m_costs[cell_i];
        }
        GRID_FOR_END
        
        for(size_t i=1;i<hsize;i++) cell_cost_hist[i] += cell_cost_hist[i-1];
        double target_cost = cell_cost_hist[hsize-1] / (*cpu_gpu_perf_ratio);

        // partiton around cell cost
        size_t i=0;
        while(i<hsize && cell_cost_hist[i]<target_cost ) ++i;
        if(i>=hsize) i=hsize-1;
        cpu_gpu_partition->m_gpu_cost_threshold = min_cost + i*(max_cost-min_cost)/hsize;

        // partition around cell index
        double cost_sum = 0.0;
        cpu_gpu_partition->cpu_cell_count = 0;
        GRID_FOR_BEGIN(grid_dims,cell_i,cell_loc)
        {
          if( cost_sum < target_cost )
          {
            cost_sum += grid_costs.m_costs[cell_i];
            ++ cpu_gpu_partition->cpu_cell_count;
          }
        }
        GRID_FOR_END
      }
      else
#     endif
      {
        cpu_gpu_partition->m_gpu_block = GridBlock{ {0,0,0} , {0,0,0} };
        cpu_gpu_partition->m_cpu_block = GridBlock{ block_start , block_end };
        cpu_gpu_partition->m_gpu_cost_threshold = max_cost;
        cpu_gpu_partition->cpu_cell_count = n_cells;
      }
      cpu_gpu_partition->gpu_cell_particles_threshold = *gpu_cell_particles_threshold;
      ldbg <<"cpu_gpu_load_balance : cpu_lb_block=" << cpu_gpu_partition->m_cpu_block 
            << " , gpu_lb_block="<< cpu_gpu_partition->m_gpu_block 
            << " , cpu_cell_count="<<cpu_gpu_partition->cpu_cell_count<<"/"<<n_cells
            << " , gpu_cost_threshold="<<cpu_gpu_partition->m_gpu_cost_threshold<<" ["<<min_cost<<";"<<max_cost<<"]\n";
    }

    struct Box1DSplit
    {
      size_t position;
      double worst_balance;
    };

    struct BoxSplit
    {
      Box1DSplit i_split;
      Box1DSplit j_split;
      Box1DSplit k_split;
    };

    inline Box1DSplit find_best_split(const std::vector<double>& values, double left_size, double right_size)
    {
      const size_t n = values.size();
      double sum_right = 0., sum_left=0.;
      
      for(double x : values) { sum_right += x; }

      if( n == 0 )
      {
        return { 0 , sum_right };
      }

      if( right_size == 0 )
      {
        return { n-1 , sum_right / n };
      }

      if( left_size == 0 )
      {
        return { 0 , sum_right / n };
      }

      double best_worst_balance = sum_right / right_size;
      size_t best_position = 0;
      for(size_t p=1 ; p<n ; p++ )
      {
        sum_left  += values[p-1];
        sum_right -= values[p-1];
        double worst_balance = std::max( sum_left/left_size , sum_right/right_size );
        if( worst_balance < best_worst_balance )
        {
          best_worst_balance = worst_balance;
          best_position = p;
        }
        //ldbg << "pos=" << p << ", best_worst_balance=" << best_worst_balance << ", worst_balance=" << worst_balance << std::endl;
      }
      return { best_position , best_worst_balance };
    }

  };


  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "cpu_gpu_load_balance", make_compatible_operator<CpuGpuLoadBalance> );
  }

}

