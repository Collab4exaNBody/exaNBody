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

#include <exanb/core/grid_algorithm.h>
#include <onika/cpp_utils.h>
#include <omp.h>


// ========== convinience macros =============

// cooperative for over a block inside a  grid (omp for only, must be inside a parallel region)
#define GRID_BLOCK_OMP_XXX_BEGIN(dims,start,end,index,loc,maindirective,pragmadirective...) { \
  const ::exanb::IJK _d = dims; \
  const ::exanb::IJK _s = start; \
  const ::exanb::IJK _e = end; \
  _Pragma(USTAMP_STR(maindirective collapse(3) pragmadirective)) \
  for( ssize_t _k=_s.k; _k<_e.k; ++ _k ) \
  for( ssize_t _j=_s.j; _j<_e.j; ++ _j ) \
  for( ssize_t _i=_s.i; _i<_e.i; ++ _i ) { \
    const ::exanb::IJK loc = {_i,_j,_k}; \
    const ssize_t index = ::exanb::grid_ijk_to_index(_d,loc); \
    (void)index;

#define GRID_BLOCK_OMP_FOR_BEGIN(dims,start,end,index,loc,pragmadirective...) GRID_BLOCK_OMP_XXX_BEGIN(dims,start,end,index,loc,omp for,pragmadirective)
#define GRID_BLOCK_OMP_TASKLOOP_BEGIN(dims,start,end,index,loc,pragmadirective...) GRID_BLOCK_OMP_XXX_BEGIN(dims,start,end,index,loc,omp taskloop firstprivate(_d) ,pragmadirective)

// future replacement
#define GRID_OMP_FOR_BEGIN(dims,index,loc,pragmadirective...) GRID_BLOCK_OMP_FOR_BEGIN(dims,::exanb::IJK(0,0,0),dims,index,loc,pragmadirective)
#define GRID_OMP_TASKLOOP_BEGIN(dims,index,loc,pragmadirective...) GRID_BLOCK_OMP_TASKLOOP_BEGIN(dims,::exanb::IJK(0,0,0),dims,index,loc,pragmadirective)

#define GRID_OMP_FOR_END }}
#define GRID_BLOCK_OMP_FOR_END GRID_OMP_FOR_END
#define GRID_OMP_TASKLOOP_END GRID_OMP_FOR_END
#define GRID_BLOCK_OMP_TASKLOOP_END GRID_OMP_FOR_END

#define GRID_OMP_FOR(dims,loc,maindirective,...) \
  _Pragma(USTAMP_STR(maindirective collapse(3) __VA_ARGS__)) \
  for( ssize_t loc##_k=0; loc##_k<dims.k; ++ loc##_k ) \
  for( ssize_t loc##_j=0; loc##_j<dims.j; ++ loc##_j ) \
  for( ssize_t loc##_i=0; loc##_i<dims.i; ++ loc##_i )

#define GRID_OMP_TASKLOOP(dims,loc,...) GRID_OMP_FOR(dims,loc,omp taskloop,__VA_ARGS__)

#include <iostream>

namespace exanb
{

  inline std::pair<ssize_t,ssize_t> integer_range_partition(ssize_t start, ssize_t end, size_t n_part, size_t part_i)
  {
    ssize_t size = end - start;
    ssize_t part_start = start + ( size * part_i ) / n_part;
    ssize_t part_end = start + ( size * (part_i+1) ) / n_part;
    return std::make_pair( part_start , part_end );
  }

  /*!
  @brief search for pairs of cells in a grid such that they may contain points within max_dist distance.
  only emits unique pairs : e.g. if (a,b) has been emited, (b,a) won't. also ensures that index(a) < index(b)
  @param dims diension of the entire containing grid
  @param start lower corner of the rectilinear block to scan 
  @param max_dist nomalized to match cell_size==1.0
  */
  template<typename CellPairOperator, typename CellStartOperator=NullCellOperator, typename CellEndOperator=NullCellOperator>
  static inline void omp_apply_grid_block_cell_pair(
    IJK dims,
    IJK start,
    IJK end, 
    double max_dist, 
    CellPairOperator func, 
    CellStartOperator begin_func=null_cell_operator,
    CellEndOperator end_func=null_cell_operator,
    bool omp_wait_opt=true )
  {
    static constexpr ssize_t zero = 0;
  
    start.k = std::max( start.k , zero );
    start.j = std::max( start.j , zero );
    start.i = std::max( start.i , zero );

    end.k = std::min( end.k , dims.k );
    end.j = std::min( end.j , dims.j );
    end.i = std::min( end.i , dims.i );
    
    double max_dist2 = max_dist * max_dist;
    ssize_t sb = std::ceil(max_dist);
    //ssize_t grid_safe_max_diff = std::floor( max_dist / std::sqrt(3.0) );

    GRID_BLOCK_OMP_FOR_BEGIN(dims,start,end,cell_i,cell, /* here after are omp for directives */ nowait )
    {
      begin_func( cell );
      
      // begin of sub block search
      // let's start with a naïve sub block search
      // search forward i. then search forward j, starting with i=[min of sub block]. then search forward k with i and j starting again from lower bound of search area
      IJK start2 = { std::max(cell.i-sb,zero) , std::max(cell.j-sb,zero) , std::max(cell.k-sb,zero) };
      IJK end2 = { std::min(cell.i+sb+1,end.i) , std::min(cell.j+sb+1,end.j) , std::min(cell.k+sb+1,end.k) };
      ssize_t k2 = cell.k;
      ssize_t j2 = cell.j;
      ssize_t i2 = cell.i+1;
      for(; k2<end2.k ; k2++)
      {
        for(; j2<end2.j ; j2++)
        {
          for(; i2<end2.i ; i2++)
          {
            // check lexicographic order
            assert( ( k2>cell.k ) || ( k2==cell.k && j2>cell.j ) || ( k2==cell.k && j2==cell.j && i2>cell.i ) );
            if( min_distance2_between( grid_cell_bounds(cell) , grid_cell_bounds(IJK{i2,j2,k2}) ) <= max_dist2 )
            {
              func( cell , IJK{i2,j2,k2} );
            }
          }
          i2 = start2.i;
        }
        j2 = start2.j;
      }
      k2 = start2.k; // useless
      // end of sub block search
      
      end_func( cell );
    }
    GRID_BLOCK_OMP_FOR_END

    if( omp_wait_opt )
    {
#   pragma omp barrier
    }
  }

  // convinience method which simply embed a call to omp_apply_grid_block_cell_pair in a parallel region
  template<typename CellPairOperator, typename CellStartOperator=NullCellOperator, typename CellEndOperator=NullCellOperator>
  static inline void parallel_apply_grid_block_cell_pair(
    IJK dims,
    IJK start,
    IJK end, 
    double max_dist, 
    CellPairOperator func, 
    CellStartOperator begin_func=null_cell_operator,
    CellEndOperator end_func=null_cell_operator)
  {
#   pragma omp parallel
    omp_apply_grid_block_cell_pair(dims,start,end,max_dist,func,begin_func,end_func, std::false_type() );
  }

}

