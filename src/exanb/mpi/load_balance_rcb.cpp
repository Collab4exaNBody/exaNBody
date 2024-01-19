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
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>

#include <exanb/core/cell_costs.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <mpi.h>
#include <exanb/mpi/all_value_equal.h>

#include <vector>
#include <string>
#include <algorithm>

#ifdef __use_lib_zoltan
#include "zoltan.h"
#endif

namespace exanb
{
  
  using std::vector;
  using std::string;
  using std::endl;

  struct LoadBalanceRCBNode : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi         , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain    , domain      , INPUT , REQUIRED );
    ADD_SLOT( GridBlock , lb_block    , INPUT_OUTPUT );
    ADD_SLOT( CellCosts , cell_costs  , INPUT , REQUIRED );
    ADD_SLOT( double    , lb_inbalance, INPUT_OUTPUT);

#   ifdef __use_lib_zoltan
    ADD_SLOT( bool       , use_zoltan             , INPUT, true );
    ADD_SLOT( std::string, zoltan_rcb_output_level, INPUT, std::string("0") );
    ADD_SLOT( std::string, zoltan_debug_level     , INPUT, std::string("0") );
    ADD_SLOT( std::string, zoltan_lb_approach     , INPUT, std::string("REPARTITION") );
    ADD_SLOT( std::string, zoltan_average_cuts    , INPUT, std::string("0") );
    ADD_SLOT( std::string, zoltan_rcb_reuse       , INPUT, std::string("1") );
#   endif

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      const CellCosts& grid_costs = *cell_costs;
      GridBlock& out_block = *lb_block;
      double& lb_inbalance = *(this->lb_inbalance);

      const IJK dom_dims = domain->grid_dimension();

      int np=1, rank=0;
      MPI_Comm_size(comm,&np);
      MPI_Comm_rank(comm,&rank);

#     ifdef __use_lib_zoltan
      if( *use_zoltan )
      {
        if( np > 1 )
        {
          if( m_zoltanStruct == nullptr )
          {
            ldbg << "RCB_REUSE        = " << *zoltan_rcb_reuse << std::endl;
            ldbg << "RCB_OUTPUT_LEVEL = " << *zoltan_rcb_output_level << std::endl;
            ldbg << "AVERAGE_CUTS     = " << *zoltan_average_cuts << std::endl;
            ldbg << "DEBUG_LEVEL      = " << *zoltan_debug_level << std::endl;
            ldbg << "LB_APPROACH      = " << *zoltan_lb_approach << std::endl;
            
            m_zoltanStruct = Zoltan_Create(comm);
            assert( m_zoltanStruct != nullptr );
            Zoltan_Set_Param(m_zoltanStruct, "RCB_REUSE", zoltan_rcb_reuse->c_str() );
            Zoltan_Set_Param(m_zoltanStruct, "RCB_OUTPUT_LEVEL", zoltan_rcb_output_level->c_str() );
            Zoltan_Set_Param(m_zoltanStruct, "AVERAGE_CUTS", zoltan_average_cuts->c_str() );
            Zoltan_Set_Param(m_zoltanStruct, "RCB_RECOMPUTE_BOX", "1");
            Zoltan_Set_Param(m_zoltanStruct, "NUM_GID_ENTRIES", "2");
            Zoltan_Set_Param(m_zoltanStruct, "NUM_LID_ENTRIES", "1");
            Zoltan_Set_Param(m_zoltanStruct, "OBJ_WEIGHT_DIM", "1"); 
            Zoltan_Set_Param(m_zoltanStruct, "KEEP_CUTS", "1");
            Zoltan_Set_Param(m_zoltanStruct, "DEBUG_LEVEL", zoltan_debug_level->c_str() );
            Zoltan_Set_Param(m_zoltanStruct, "LB_APPROACH", zoltan_lb_approach->c_str() );
            Zoltan_Set_Param(m_zoltanStruct, "LB_METHOD", "RCB");
          }
          /*else
          {
            Zoltan_Set_Param(m_zoltanStruct, "LB_APPROACH", "REPARTITION");
          }*/
          
          m_num_obj_fn = [&grid_costs](int* ierr)
            -> int
            {
              IJK dims = dimension( grid_costs.m_block );
              size_t n_cells = grid_cell_count( dims );
              ierr = 0;
              return n_cells;
            };
          Zoltan_Set_Num_Obj_Fn(m_zoltanStruct, zoltan_num_obj, &m_num_obj_fn );
          
          m_obj_list_fn = [dom_dims,rank,&grid_costs](int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr)
            -> void
            {
              assert( wgt_dim == 1 );
              assert( num_gid_entries == 2 );
              assert( num_lid_entries == 1 );
              IJK dims = dimension( grid_costs.m_block );
#             pragma omp parallel
              {
                GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(static) )
                {
                  size_t global_i = grid_ijk_to_index( dom_dims , loc + grid_costs.m_block.start );
                  local_ids[i] = i;
                  global_ids[i*2+0] = global_i;
                  global_ids[i*2+1] = rank;
                  obj_wgts[i] = grid_costs.m_costs[i];
                }
                GRID_OMP_FOR_END
              }
              ierr=0;
            };
          Zoltan_Set_Obj_List_Fn(m_zoltanStruct, zoltan_obj_list, &m_obj_list_fn );

          m_geom_dim_fn = [](int* ierr) ->int { return 3; };
          Zoltan_Set_Num_Geom_Fn(m_zoltanStruct, zoltan_geom_dim, &m_geom_dim_fn);

          m_geom_fn = [this,dom_dims](int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, double *geom_vec, int *ierr)
            -> void
            {
              assert( num_gid_entries == 2 );
              assert( num_lid_entries == 1 );
              IJK dom_loc = grid_index_to_ijk( dom_dims , *global_id );
              Vec3d r = (dom_loc*1.0) ; //+ 0.5;
              // ldbg << "geom="<<r<<std::endl;
              geom_vec[0] = r.x;
              geom_vec[1] = r.y;
              geom_vec[2] = r.z;
              ierr = 0;
            };
          Zoltan_Set_Geom_Fn(m_zoltanStruct, zoltan_geom, &m_geom_fn );        

          int changes=0, numGidEntries=0, numLidEntries=0, numImport=0, numExport=0;
          ZOLTAN_ID_PTR importGlobalGids=nullptr, importLocalGids=nullptr, exportGlobalGids=nullptr, exportLocalGids=nullptr; 
          int *importProcs=nullptr, *importToPart=nullptr, *exportProcs=nullptr, *exportToPart=nullptr;

          int zerr = Zoltan_LB_Partition(
            m_zoltanStruct,
            &changes, /* Flag indicating whether partition changed */
            &numGidEntries,
            &numLidEntries,
            &numImport, /* objects to be imported to new part */
            &importGlobalGids, 
            &importLocalGids,
            &importProcs, 
            &importToPart,
            &numExport, /* # objects to be exported from old part */
            &exportGlobalGids, 
            &exportLocalGids, 
            &exportProcs, 
            &exportToPart);

          ldbg <<"zoltan err = "<<zerr<<std::endl;

          int ndim = 3;
          double xmin = dom_dims.i;
          double ymin = dom_dims.j;
          double zmin = dom_dims.k;
          double xmax = 0.;
          double ymax = 0.;
          double zmax = 0.;
          Zoltan_RCB_Box(m_zoltanStruct,rank,&ndim,&xmin,&ymin,&zmin,&xmax,&ymax,&zmax);
          
          xmin = std::max( 0.0 , xmin );
          ymin = std::max( 0.0 , ymin );
          zmin = std::max( 0.0 , zmin );
          xmax = std::min( dom_dims.i*1.0 , xmax );
          ymax = std::min( dom_dims.j*1.0 , ymax );
          zmax = std::min( dom_dims.k*1.0 , zmax );
                
          ldbg << "RCB: min="<<xmin<<','<<ymin<<','<<zmin<<" max="<<xmax<<','<<ymax<<','<<zmax<<" ndim="<<ndim<<std::endl<<std::flush;
          
          out_block.start = IJK { static_cast<ssize_t>(std::floor(xmin+0.5)) , static_cast<ssize_t>(std::floor(ymin+0.5)) , static_cast<ssize_t>(std::floor(zmin+0.5)) };
          out_block.end = IJK { static_cast<ssize_t>(std::floor(xmax+0.5)) , static_cast<ssize_t>(std::floor(ymax+0.5)) , static_cast<ssize_t>(std::floor(zmax+0.5)) };
          
          out_block.start.i = std::max( out_block.start.i , 0l );
          out_block.start.j = std::max( out_block.start.j , 0l );
          out_block.start.k = std::max( out_block.start.k , 0l );
          out_block.end.i = std::min( out_block.end.i , dom_dims.i );
          out_block.end.j = std::min( out_block.end.j , dom_dims.j );
          out_block.end.k = std::min( out_block.end.k , dom_dims.k );

          lb_inbalance = 0.1; // lack real computation
        }
        else
        {
          out_block.start = IJK { 0, 0, 0 };
          out_block.end = dom_dims;
        }
        ldbg <<"Zoltan based RCB : block = "<<out_block<<std::endl<<std::flush;
        //MPI_Barrier( comm );
      }
      else
#     endif
      {    

        size_t domain_n_cells = grid_cell_count( domain->grid_dimension() );
        IJK block_start = { 0, 0, 0 };
        IJK block_end = domain->grid_dimension();
	
        CellCosts all_grid_costs;
        all_grid_costs.m_block = GridBlock{ block_start , block_end };
        all_grid_costs.m_costs.resize( domain_n_cells , 0.0 );

        if( dom_dims != domain->grid_dimension() )
        {
          fatal_error() << "grid costs size incompatible with domain size. dom_dims="<<domain->grid_dimension()<<", block="<<all_grid_costs.m_block<<std::endl;
        }
        
        IJK grid_dims = dimension(grid_costs.m_block);
        //assert( grid_dims == grid.dimension() );

        ldbg << "---- LoadBalanceRCBNode: domain=" << block_end  << " ----" << std::endl;
        unsigned long nb_ignored_grid_cells = 0;
        GRID_FOR_BEGIN(grid_dims,i,loc)
        {
          auto dom_loc = loc + grid_costs.m_block.start;
          if( grid_contains( dom_dims , dom_loc ) )
          {
            ssize_t j = grid_ijk_to_index( dom_dims, dom_loc );
            assert( i >= 0 && i < static_cast<ssize_t>(grid_costs.m_costs.size()) );
            assert( j >= 0 && j < static_cast<ssize_t>(all_grid_costs.m_costs.size()) );
            all_grid_costs.m_costs[j] = grid_costs.m_costs[i];
          }
          else if( grid_costs.m_costs[i] > 0.0 )
          {
            ++ nb_ignored_grid_cells;
          }
        }
        GRID_FOR_END

#       ifndef NDEBUG
        MPI_Allreduce(MPI_IN_PLACE,&nb_ignored_grid_cells,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
#       endif
        if( nb_ignored_grid_cells > 0 )
        {
          lerr << "Warning: "<<nb_ignored_grid_cells<<" grid cells with positive costs ignored (out of domain)"<<std::endl;
        }
        
        MPI_Allreduce(MPI_IN_PLACE,all_grid_costs.m_costs.data(),domain_n_cells,MPI_DOUBLE,MPI_SUM,comm);

        IJK block_dims = block_end - block_start;
        std::vector<double> i_cost( block_dims.i , 0. );
        std::vector<double> j_cost( block_dims.j , 0. );
        std::vector<double> k_cost( block_dims.k , 0. );

        assert( block_dims == dom_dims );
        GRID_FOR_BEGIN(block_dims,cell_i,cell_loc)
        {
	        i_cost[ cell_loc.i ] += all_grid_costs.m_costs[cell_i];
	        j_cost[ cell_loc.j ] += all_grid_costs.m_costs[cell_i];
	        k_cost[ cell_loc.k ] += all_grid_costs.m_costs[cell_i];
        }
        GRID_FOR_END
        
        int group_size = np;
        int rank_in_group = rank;
        int level = 0;
        while( group_size > 1 && !is_empty( GridBlock{block_start,block_end} ) )
        {
          assert( rank_in_group>=0 && rank_in_group<group_size );

  //        ldbg << "level="<<level<<", rankInGroup="<<rank_in_group<<", groupSize="<<group_size<<", block_start="<<block_start<<", block_end="<<block_end<< std::endl;

          block_dims = block_end - block_start;

          i_cost.assign( block_dims.i , 0. );
          j_cost.assign( block_dims.j , 0. );
          k_cost.assign( block_dims.k , 0. );
          GRID_FOR_BEGIN(block_dims,_,cell_loc)
          {
            size_t grid_cell_index = grid_ijk_to_index( dom_dims, cell_loc + block_start );
            assert( cell_loc.i>=0 && cell_loc.i < static_cast<ssize_t>(i_cost.size()) );
            assert( cell_loc.j>=0 && cell_loc.j < static_cast<ssize_t>(j_cost.size()) );
            assert( cell_loc.k>=0 && cell_loc.k < static_cast<ssize_t>(k_cost.size()) );
	          i_cost[ cell_loc.i ] += all_grid_costs.m_costs[grid_cell_index];
	          j_cost[ cell_loc.j ] += all_grid_costs.m_costs[grid_cell_index];
	          k_cost[ cell_loc.k ] += all_grid_costs.m_costs[grid_cell_index];
          }
          GRID_FOR_END

  //        ldbg << "Hist I : "; print_histogram(i_cost);
  //        ldbg << "Hist J : "; print_histogram(j_cost);
  //        ldbg << "Hist K : "; print_histogram(k_cost);


		      int left_size = group_size/2;
		      int right_size = group_size - left_size;
		      int next_level_rank = rank_in_group;
          bool next_level_side = (rank_in_group >= left_size);
          int next_level_size = left_size;

          if( next_level_side )
          {
	          next_level_rank = rank_in_group - left_size;
	          next_level_size = group_size - left_size;
          }

          assert( next_level_rank>=0 && next_level_rank<next_level_size );

          Box1DSplit split[3];
          
          split[0] = find_best_split(i_cost,left_size,right_size );
          assert( split[0].position >= 0 && static_cast<ssize_t>(split[0].position) < block_dims.i );
          split[0].surf = block_dims.j * block_dims.k;
          split[0].valid = ( block_dims.i >= 2 ) && ( split[0].position>0 ) && ( static_cast<ssize_t>(split[0].position) < block_dims.i );
          split[0].axis = 0;
          
          split[1] = find_best_split(j_cost,left_size,right_size );
          assert( split[1].position >= 0 && static_cast<ssize_t>(split[1].position) < block_dims.j );
          split[1].surf = block_dims.i * block_dims.k;
          split[1].valid = ( block_dims.j >= 2 ) && ( split[1].position>0 ) && ( static_cast<ssize_t>(split[1].position) < block_dims.j );
          split[1].axis = 1;

          split[2] = find_best_split(k_cost,left_size,right_size );
          assert( split[2].position >= 0 && static_cast<ssize_t>(split[2].position) < block_dims.k );
          split[2].surf = block_dims.i * block_dims.j;
          split[2].valid = ( block_dims.k >= 2 ) && ( split[2].position>0 ) && ( static_cast<ssize_t>(split[2].position) < block_dims.k );
          split[2].axis = 2;
          
          auto split_better_than = [](const Box1DSplit& a, const Box1DSplit& b) -> bool
          {
            if( ! a.valid ) return false;
            if( ! b.valid ) return true;
            double max_wb = std::max( a.worst_balance , b.worst_balance );
            if( max_wb == 0.0 ) return true;
            double min_wb = std::min( a.worst_balance , b.worst_balance );
            if( min_wb / max_wb > 0.95 ) return ( a.surf < b.surf );
            else return ( a.worst_balance < b.worst_balance );
          };
          
          std::sort( split , split+3 , split_better_than );
	  
          GridBlock left_block = { block_start , block_end };
          GridBlock right_block = { block_start , block_end };

          char split_choice = '?';
          if( split[0].valid )
          {
            if( split[0].axis == 0 )
            {
              split_choice = 'I';
              left_block.end.i = block_start.i + split[0].position;
              right_block.start.i = block_start.i + split[0].position;
            }
            else if( split[0].axis == 1 )
            {
              split_choice = 'J';
              left_block.end.j = block_start.j + split[0].position;
              right_block.start.j = block_start.j + split[0].position;
            }
            else if( split[0].axis == 2 )
            {
              split_choice = 'K';
              left_block.end.k = block_start.k + split[0].position;
              right_block.start.k = block_start.k + split[0].position;
            }
          }
          else
          {
            ldbg << "impossible partitioning, splitting regardless of costs : dims="<<block_dims<< std::endl;
            if( block_dims.i >= block_dims.j && block_dims.i >= block_dims.k )
            {
              split_choice = 'I';
              assert( block_dims.i >= 2 );
              left_block.end.i = block_start.i + block_dims.i/2;
              right_block.start.i = left_block.end.i;
            }
            else if( block_dims.j >= block_dims.i && block_dims.j >= block_dims.k )
            {
              split_choice = 'J';
              assert( block_dims.j >= 2 );
              left_block.end.j = block_start.j + block_dims.j/2;
              right_block.start.j = left_block.end.j;
            }
            else
            {
              split_choice = 'K';
              assert( block_dims.k >= 2 );
              left_block.end.k = block_start.k + block_dims.k/2;
              right_block.start.k = left_block.end.k;
            }
          }
          assert( ! is_empty(left_block) );
          assert( ! is_empty(right_block) );

          ldbg << "split level "<<level
               <<" : [a="<<split[0].axis<<";p="<<split[0].position<<";wb="<<split[0].worst_balance<<";s="<<split[0].surf<<";v="<<split[0].valid<<"]"
               <<" , [a="<<split[1].axis<<";p="<<split[1].position<<";wb="<<split[1].worst_balance<<";s="<<split[1].surf<<";v="<<split[1].valid<<"]"
               <<" , [a="<<split[2].axis<<";p="<<split[2].position<<";wb="<<split[2].worst_balance<<";s="<<split[2].surf<<";v="<<split[2].valid<<"]"
               <<" -> "<< split_choice <<std::endl;

          //ldbg << "left  block = " << left_block << std::endl;
          //ldbg << "right block = " << right_block << std::endl;

          GridBlock next_block = left_block;
          if( next_level_side )
	        {
	          next_block = right_block;
         	}
          //ldbg << "next_block = " << next_block << std::endl;
       	
        	group_size = next_level_size;
        	rank_in_group = next_level_rank;
        	++ level;

	        block_start = next_block.start;
	        block_end = next_block.end;
        }

        // final block allocated to this processor
        out_block = GridBlock{ block_start , block_end };
        
        // count cost in this block to get inbalance feedback
        double out_block_cost = 0.0;
        block_dims = block_end - block_start;
        GRID_FOR_BEGIN(block_dims,_,cell_loc)
        {
          size_t grid_cell_index = grid_ijk_to_index( dom_dims, cell_loc + block_start );
          out_block_cost += all_grid_costs.m_costs[grid_cell_index];
        }    
        GRID_FOR_END

        // get estimated inbalance accross all processors
        std::vector<double> all_cost(np,0.0);
        MPI_Allgather(&out_block_cost,1,MPI_DOUBLE,all_cost.data(),1,MPI_DOUBLE,comm);
        double avg_cost = 0.0;
        double max_cost = 0.0;
        for( double x : all_cost ) { avg_cost += x; max_cost = std::max(x,max_cost); }
        avg_cost /= np;
        lb_inbalance = 0.0;
        if( avg_cost > 0.0 ) { lb_inbalance = (max_cost-avg_cost)/avg_cost ; }

        ldbg << "cost="<<out_block_cost<<", max="<<max_cost<<", avg="<<avg_cost << std::endl;
      }
      
      // it should'nt be an error, though it has not been tested.
      if( is_empty(out_block) )
      {
        fatal_error() << "Assigned grid block is empty : domain="<< (*domain)<<" , block="<<out_block << std::endl;
      }

      if( out_block.start.i<0 || out_block.start.j<0 || out_block.start.k<0 || out_block.end.i>dom_dims.i || out_block.end.j>dom_dims.j || out_block.end.k>dom_dims.k )
      {
        fatal_error() << "Assigned grid block doesn't fit in domain's grid : domain="<<(*domain)<<" , block="<<out_block <<std::endl;
      }

      ldbg << "domain = { "<<(*domain)<< " } , out_block={" << out_block << "} , inb="<< lb_inbalance << std::endl << std::flush;
    }

    struct Box1DSplit
    {
      size_t position = 0;
      double worst_balance = 0.0;
      long surf = 0;
      int axis = 0;
      bool valid = false;
    };

    inline void print_histogram(std::vector<double>& values)
    {
      ldbg << "[" ;
      for(double x : values)
      {
        ldbg << format_string(" %.1e",x);
      }
      ldbg << " ]" << std::endl;
    }

    inline Box1DSplit find_best_split(const std::vector<double>& values, size_t left_size, size_t right_size /*, size_t ghost_layers */ )
    {    
      size_t n = values.size();
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
      return { best_position , best_worst_balance , -1 , -1 , false };
    }

  private:  
#   ifdef __use_lib_zoltan
    Zoltan_Struct* m_zoltanStruct = nullptr;

    std::function<int(int*)> m_num_obj_fn;
    static inline int zoltan_num_obj(void *data, int *ierr)
    {
      auto* fn = reinterpret_cast< std::function<int(int*)> * >( data );
      return (*fn) (ierr);
    }

    std::function<void(int,int,ZOLTAN_ID_PTR,ZOLTAN_ID_PTR,int,float*,int*)> m_obj_list_fn;
    static void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr)
    {
      auto* fn = reinterpret_cast< std::function<void(int,int,ZOLTAN_ID_PTR,ZOLTAN_ID_PTR,int,float*,int*)> * >( data );
      (*fn) (num_gid_entries,num_lid_entries,global_ids,local_ids,wgt_dim,obj_wgts,ierr);
    }

    std::function<int(int*)> m_geom_dim_fn;
    static inline int zoltan_geom_dim(void *data, int *ierr)
    {
      auto* fn = reinterpret_cast< std::function<int(int*)> * >( data );
      return (*fn) (ierr);
    }

    std::function<void(int,int,ZOLTAN_ID_PTR,ZOLTAN_ID_PTR,double*,int*)> m_geom_fn;
    static inline void zoltan_geom (void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, double *geom_vec, int *ierr)
    {
      auto* fn = reinterpret_cast< std::function<void(int,int,ZOLTAN_ID_PTR,ZOLTAN_ID_PTR,double*,int*)> * >( data );
      (*fn) (num_gid_entries,num_lid_entries,global_id,local_id,geom_vec,ierr);
    }
#   endif
  };


  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
#   ifdef __use_lib_zoltan
    static float version = 0.0;
    static int argc = 1;
    char arg0[] = "ExaStampV2\0";
    static char* argv[1] = { arg0 };
    Zoltan_Initialize(argc, argv, &version);
    ldbg << "Using zoltan v"<<version<<std::endl;
#   endif

    OperatorNodeFactory::instance()->register_factory( "load_balance_rcb", make_compatible_operator<LoadBalanceRCBNode> );
  }

}

