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

#include <exanb/core/basic_types.h>
#include <exanb/core/geometry.h>
#include <cmath>
#include <algorithm>


#define XSTAMP_IGNORE_UNUSED_VARIABLE(x) if(false){auto y=x;x=y;}

// convinience macros
#define GRID_FOR_BEGIN(dims,index,cell) { \
  ::exanb::IJK _d=dims; \
  ssize_t index=0; XSTAMP_IGNORE_UNUSED_VARIABLE(index) \
  for(ssize_t _k=0;_k<_d.k;++_k) \
  for(ssize_t _j=0;_j<_d.j;++_j) \
  for(ssize_t _i=0;_i<_d.i;++_i,++index) { \
    ::exanb::IJK cell={_i,_j,_k}; XSTAMP_IGNORE_UNUSED_VARIABLE(cell)

#define GRID_FOR_END } }

// loop over a block inside a  grid 
#define GRID_BLOCK_FOR_BEGIN(dims,start,end,index,loc) { \
  ::exanb::IJK _d = dims; \
  ::exanb::IJK _s = start; \
  ::exanb::IJK _e = end; \
  for(ssize_t _k=_s.k;_k<_e.k;_k++) { \
  for(ssize_t _j=_s.j;_j<_e.j;_j++) { \
  for(ssize_t _i=_s.i;_i<_e.i;_i++) { \
    ::exanb::IJK loc = ::exanb::IJK{_i,_j,_k}; \
    ssize_t index = ::exanb::grid_ijk_to_index(_d,loc); XSTAMP_IGNORE_UNUSED_VARIABLE(index)

#define GRID_BLOCK_FOR_END }}} }

// convinience macro when all you do is calling a function
#define APPLY_GRID(dims,func) GRID_FOR_BEGIN(dims,_cell_index,_cell_loc) func(_cell_loc); GRID_FOR_END

// convinience macro when all you do is calling a function
#define APPLY_GRID_BLOCK(dims,start,end,func) GRID_BLOCK_FOR_BEGIN(dims,start,end,_cell_index,_cell_loc) func(_cell_index,_cell_loc); GRID_BLOCK_FOR_END


namespace exanb
{
  // cell operator defaults to null operator. this is convinient for begin and end operator in grid_apply_cell_pair
# if __cplusplus >= 201701
  inline constexpr void null_cell_operator(IJK){}
# else
  inline void null_cell_operator(IJK){}
# endif
  using NullCellOperator = decltype(null_cell_operator);

  ONIKA_HOST_DEVICE_FUNC inline IJK dimension(const GridBlock& gb)
  {
    return IJK{ gb.end.i-gb.start.i , gb.end.j-gb.start.j , gb.end.k-gb.start.k };
  }

  // number of cells in a grid
  ONIKA_HOST_DEVICE_FUNC inline ssize_t grid_cell_count(const IJK& dims)
  {
    return dims.i * dims.j * dims.k;
  }
  
//  [[deprecated]]
//  static inline ssize_t grid_count(IJK dims) { return grid_cell_count(dims); }

  // computes grid element index from its (i,j,k) coordinates, given that the grid size is dims
  // assumes elements are ordered according to the lexicographic order of (k,j,i)
  ONIKA_HOST_DEVICE_FUNC inline ssize_t grid_ijk_to_index(const IJK& dims, const IJK& p)
  {
    ssize_t index = p.k;
    index *= dims.j;
    index += p.j;
    index *= dims.i;
    index += p.i;
    return index;
  }

  // loc is relative (to grid) IJK coordinate.
  ONIKA_HOST_DEVICE_FUNC inline bool grid_contains(const IJK& dims, const IJK& loc)
  {
    return loc.i>=0 && loc.i<dims.i
        && loc.j>=0 && loc.j<dims.j
        && loc.k>=0 && loc.k<dims.k ;
  }

  // computes grid element index from its (i,j,k) coordinates, given that the grid size is dims
  // assumes elements are ordered according to the lexicographic order of (k,j,i)
  ONIKA_HOST_DEVICE_FUNC inline IJK grid_index_to_ijk(const IJK& dims, ssize_t index)
  {
    ssize_t i = index % dims.i;
    index /= dims.i;
    ssize_t j = index % dims.j;
    index /= dims.j;
    ssize_t k = index;
    return IJK{i,j,k};
  }

  inline IJK min (const IJK& a, const IJK& b)
  {
    return IJK{ std::min(a.i,b.i) , std::min(a.j,b.j) , std::min(a.k,b.k) };
  }

  inline IJK max (const IJK& a, const IJK& b)
  {
    return IJK{ std::max(a.i,b.i) , std::max(a.j,b.j) , std::max(a.k,b.k) };
  }

  inline GridBlock intersection(const GridBlock& a, const GridBlock& b)
  {
    return GridBlock{ max(a.start,b.start) , min(a.end,b.end) };
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_empty(const GridBlock& b)
  {
    return b.start.i>=b.end.i || b.start.j>=b.end.j || b.start.k>=b.end.k;
  }

  // traverse the entire grid except cells that are up to margin cells from border
  // i.e. margin=1 means all but outermost cells
  template<typename CellOperator>
  inline void apply_grid_interior(const IJK& dims, ssize_t margin, CellOperator cell_func)
  {
    APPLY_GRID_BLOCK( dims, IJK(margin,margin,margin) , IJK(dims.i-margin,dims.j-margin,dims.k-margin) , cell_func );
  }
  
  ONIKA_HOST_DEVICE_FUNC inline bool inside_grid_shell(const IJK& dims, ssize_t margin, ssize_t thickness,const IJK& cell)
  {
    using onika::cuda::min;
    ssize_t di = min( cell.i , (dims.i-1-cell.i) );
    ssize_t dj = min( cell.j , (dims.j-1-cell.j) );
    ssize_t dk = min( cell.k , (dims.k-1-cell.k) );
    ssize_t b = min( min(di,dj) , dk ); // b is the distance to the nearest border
    return ( b>=margin && b<(margin+thickness) );
  }
  
  ONIKA_HOST_DEVICE_FUNC inline bool inside_block(GridBlock block, IJK p)
  {
    return p.i>=block.start.i && p.i<block.end.i &&
           p.j>=block.start.j && p.j<block.end.j &&
           p.k>=block.start.k && p.k<block.end.k;
  }

  ONIKA_HOST_DEVICE_FUNC inline GridBlock enlarge_block(GridBlock block, ssize_t n)
  {
    return { block.start-n , block.end+n };
  }
  
  // traverse a shell of cells, with thickness, at margin from the border.
  // apply cell_func to all grid cells 'cell' such that inside_grid_shell(dim,margin,thickness,cell) == true
  // i.e thickness=1, margin=0 is the outer most layer of cells
  template<class CellOperator>
  inline void apply_grid_shell(const IJK& dims, ssize_t margin, ssize_t thickness, CellOperator cell_func)
  {
    assert( margin >= 0 );
    assert( thickness >= 0 );
  
    // no cell is away enough from the border
    if( dims.i<=(2*margin) || dims.j<=(2*margin) || dims.k<=(2*margin))
    {
      return;
    }
  
    IJK start1 = IJK{0,0,0} + margin;
    IJK end1 = start1 + thickness;

    IJK start2 = ( dims - margin ) - thickness;
    //IJK end2 = dims - margin;
  
    // we have a single contiguous block (no central hole)
    if( end1.i>start2.i || end1.j>start2.j || end1.j>start2.j )
    {
      // equivalent to apply_grid_interior
      APPLY_GRID_BLOCK( dims, IJK(margin,margin,margin) , IJK(dims.i-margin,dims.j-margin,dims.k-margin) , cell_func );      
    }
    else
    {
      // Z planes
      APPLY_GRID_BLOCK(  dims, IJK(margin,margin,margin) , IJK(dims.i-margin,dims.j-margin,margin+thickness) , cell_func );   
      APPLY_GRID_BLOCK(  dims, IJK(margin,margin,std::max(margin+thickness,dims.k-margin-thickness)) , IJK(dims.i-margin,dims.j-margin,dims.k-margin) , cell_func );

      // Y planes
      APPLY_GRID_BLOCK(  dims, IJK(margin,margin,margin+thickness) , IJK(dims.i-margin,margin+thickness,dims.k-margin-thickness) , cell_func );
      APPLY_GRID_BLOCK(  dims, IJK(margin,std::max(margin+thickness,dims.j-margin-thickness),margin+thickness) , IJK(dims.i-margin,dims.j-margin,dims.k-margin-thickness) , cell_func );

      // X planes
      APPLY_GRID_BLOCK(  dims, IJK(margin,margin+thickness,margin+thickness) , IJK(margin+thickness,dims.j-margin-thickness,dims.k-margin-thickness) , cell_func );
      APPLY_GRID_BLOCK(  dims, IJK(std::max(margin+thickness,dims.i-margin-thickness),margin+thickness,margin+thickness) , IJK(dims.i-margin,dims.j-margin-thickness,dims.k-margin-thickness) , cell_func );
    }
  }



  // 2 types of search : pairs of cells, such that they are within a certain distance.
  // or : cells such that max distance from an external (disjoint) AABB is within a certain distance.
  // should enforce ordering of pairs : only find pairs of cells (a,b) such that index(a) < index(b)

  /*!
  @brief search for pairs of cells in a grid such that they may contain points within max_dist distance.
  only emits unique pairs : e.g. if (a,b) has been emited, (b,a) won't. also ensures that index(a) < index(b)
  @param dims diension of the entire containing grid
  @param start lower corner of the rectilinear block to scan 
  @param max_dist nomalized to match cell_size==1.0
  */
  template<typename CellPairOperator, typename CellStartOperator=NullCellOperator, typename CellEndOperator=NullCellOperator>
  inline void apply_grid_block_cell_pair(
    IJK dims,
    IJK start,
    IJK end, 
    double max_dist, 
    CellPairOperator func, 
    CellStartOperator begin_func=null_cell_operator,
    CellEndOperator end_func=null_cell_operator)
  {
    assert( start.i >= 0 );
    assert( start.j >= 0 );
    assert( start.k >= 0 );

    assert( end.i <= dims.i );
    assert( end.j <= dims.j );
    assert( end.k <= dims.k );
    
    double max_dist2 = max_dist * max_dist;
    ssize_t sb = std::ceil(max_dist);
    
    for(ssize_t k=start.k; k<end.k; k++)
    {
      for(ssize_t j=start.j; j<end.j; j++)
      {
        for(ssize_t i=start.i; i<end.i; i++)
        {
          begin_func( IJK{i,j,k} );
          
          // begin of sub block search
          // let's start with a naïve sub block search
          // search forward i. then search forward j, starting with i=[min of sub block]. then search forward k with i and j starting again from lower bound of search area
          IJK start2 = { std::max(i-sb,0l) , std::max(j-sb,0l) , std::max(k-sb,0l) };
          IJK end2 = { std::min(i+sb+1,end.i) , std::min(j+sb+1,end.j) , std::min(k+sb+1,end.k) };
          ssize_t k2 = k;
          ssize_t j2 = j;
          ssize_t i2 = i+1;
          for(; k2<end2.k ; k2++)
          {
            ssize_t dk = std::max( k2-k-1 , 0l );
            dk *= dk;
            for(; j2<end2.j ; j2++)
            {
              ssize_t dj = std::max( std::abs(j2-j)-1 , 0l );
              dj *= dj;
              for(; i2<end2.i ; i2++)
              {
                // check lexicographical order
                assert( ( k2>k ) || ( k2==k && j2>j ) || ( k2==k && j2==j && i2>i ) );
                ssize_t di = std::max( std::abs(i2-i)-1 , 0l );
                di *= di;
                ssize_t d2 = di+dj+dk;
#               ifndef NDEBUG
                double md2b = min_distance2_between( grid_cell_bounds(IJK{i,j,k}) , grid_cell_bounds(IJK{i2,j2,k2}) );
                assert( d2 == md2b );
#               endif
                if( d2 <= max_dist2 )
                {
                  func( IJK{i,j,k} , IJK{i2,j2,k2} );
                }
              }
              i2 = start2.i;
            }
            j2 = start2.j;
          }
          k2 = start2.k; // useless
          // end of sub block search
          
          end_func( IJK{i,j,k} );
        }
      }
    }    

  }
 

  /*!
  same as previous version, except :
    - no end_func operator
    - func_full is called for cell pairs (c1,c2) such that max_distance_between(c1,c2)<max_dist
    - func_partial is called for cell pairs (c1,c2) such that min_distance_between(c1,c2)<max_dist
  */
  template<typename CellPairOperatorFull, typename CellPairOperatorPartial, typename CellStartOperator=NullCellOperator>
  inline void apply_grid_block_cell_pair_fp(
    IJK dims,
    IJK start,
    IJK end, 
    double max_dist, 
    CellPairOperatorFull func_full,
    CellPairOperatorPartial func_partial,
    CellStartOperator begin_func=null_cell_operator)
  {
    assert( start.i >= 0 );
    assert( start.j >= 0 );
    assert( start.k >= 0 );

    assert( end.i <= dims.i );
    assert( end.j <= dims.j );
    assert( end.k <= dims.k );
    
    double max_dist2 = max_dist * max_dist;
    ssize_t sb = std::ceil(max_dist);
    
    for(ssize_t k=start.k; k<end.k; k++)
    {
      for(ssize_t j=start.j; j<end.j; j++)
      {
        for(ssize_t i=start.i; i<end.i; i++)
        {
          begin_func( IJK{i,j,k} );
          
          // begin of sub block search
          // let's start with a naïve sub block search
          // search forward i. then search forward j, starting with i=[min of sub block]. then search forward k with i and j starting again from lower bound of search area
          IJK start2 = { std::max(i-sb,0l) , std::max(j-sb,0l) , std::max(k-sb,0l) };
          IJK end2 = { std::min(i+sb+1,end.i) , std::min(j+sb+1,end.j) , std::min(k+sb+1,end.k) };
          ssize_t k2 = k;
          ssize_t j2 = j;
          ssize_t i2 = i+1;
          for(; k2<end2.k ; k2++)
          {
            ssize_t dkdiff = std::abs( k2-k );
            ssize_t dkmin = std::max( dkdiff - 1 , 0l );
            ssize_t dkmax = dkdiff + 1;
            dkmin *= dkmin;
            dkmax *= dkmax;

            for(; j2<end2.j ; j2++)
            {
              ssize_t djdiff = std::abs(j2-j);
              ssize_t djmin = std::max( djdiff - 1 , 0l );
              ssize_t djmax = djdiff + 1;
              djmin *= djmin;
              djmax *= djmax;
              for(; i2<end2.i ; i2++)
              {
                // check lexicographical order
                assert( ( k2>k ) || ( k2==k && j2>j ) || ( k2==k && j2==j && i2>i ) );
                ssize_t didiff = std::abs(i2-i);
                ssize_t dimax = didiff + 1;
                dimax *= dimax;
                ssize_t d2max = dimax+djmax+dkmax;                
                assert( d2max == max_distance2_between( grid_cell_bounds(IJK{i,j,k}) , grid_cell_bounds(IJK{i2,j2,k2}) ) );

                if( d2max <= max_dist2 )
                {
                  func_full( IJK{i,j,k} , IJK{i2,j2,k2} );
                }
                else
                {
                  ssize_t dimin = std::max( didiff - 1 , 0l );
                  dimin *= dimin;
                  ssize_t d2min = dimin+djmin+dkmin;                
                  assert( d2min == min_distance2_between( grid_cell_bounds(IJK{i,j,k}) , grid_cell_bounds(IJK{i2,j2,k2}) ) );
                  if( d2min <= max_dist2 )
                  {
                    func_partial( IJK{i,j,k} , IJK{i2,j2,k2} );
                  }
                }
              }
              i2 = start2.i;
            }
            j2 = start2.j;
          } // end of sub block search     

        }
      }
    }    

  }


  /*!
  @brief search for all cell C a grid which size is given by dims, where there exist point Pa inside ref and there exist a point Pb inside C such that |Pa,Pb| <= max_dist
  // WARNING: ref and max_dist are expressed in local grid frame of reference : starts at (0,0,0) and each cell's size is 1x1x1
  */
  template<typename CellOperator>
  inline void apply_grid_cell_nearby(IJK dims, AABB ref, double max_dist, CellOperator func)
  {
    static constexpr ssize_t zero = 0;

    // computes absolute block that may contain points within given distance
    IJK start = { static_cast<ssize_t>(std::floor(ref.bmin.x-max_dist)) , static_cast<ssize_t>(std::floor(ref.bmin.y-max_dist)) , static_cast<ssize_t>(std::floor(ref.bmin.z-max_dist)) };
    IJK end = { static_cast<ssize_t>(std::ceil(ref.bmax.x+max_dist)) , static_cast<ssize_t>(std::ceil(ref.bmax.y+max_dist)) , static_cast<ssize_t>(std::ceil(ref.bmax.z+max_dist)) };

    start.k = std::max( start.k , zero );
    start.j = std::max( start.j , zero );
    start.i = std::max( start.i , zero );

    end.k = std::min( end.k , dims.k );
    end.j = std::min( end.j , dims.j );
    end.i = std::min( end.i , dims.i );

    double max_dist2 = max_dist * max_dist;

    // TODO: high priority: do not compute min_distance_between for pairs you know in advance the distance will be less than max_dist
    // TODO: low priority: verify (prove or deny) that covered area is convex, then optimize cells that are tested with min_distance_between
    for(ssize_t k=start.k; k<end.k; k++)
    {
      for(ssize_t j=start.j; j<end.j; j++)
      {
        for(ssize_t i=start.i; i<end.i; i++)
        {
          double min_dist2 = min_distance2_between( ref , grid_cell_bounds(IJK{i,j,k}) );
          if( min_dist2 < max_dist2 )
          {
            func( IJK{i,j,k} );
          }
        }
      } 
    }
  }


  // 2 distinct cells, cell_a and cell_b, each subdivided into sub grid cells.
  // cell_a is divided into a cubic grid of sgside_a^3 sub cells 
  // cell_b is divided into a cubic grid of sgside_b^3 sub cells
  // we search for pair of sub grid cells, with one sub grid cell in cell_a and one in cell_b, such that the min distance between those
  // 2 sub grid cells is less or equal to max_dist.
  // max_dist is relative to sizes of cell_a and cell_b which are understood as unit cubes of size 1x1x1. e.g. max_dist=0.5 means half of the size of cell_[a|b]
  template<typename CellOperator>
  inline void apply_inter_cell_sub_grid_cell_pair(IJK cell_a, int sgside_a, IJK cell_b, int sgside_b, double max_dist, CellOperator func)
  {
    IJK rel_cell_b = cell_b - cell_a;
    AABB rel_bounds_b = grid_cell_bounds( rel_cell_b );
    rel_bounds_b.bmin = rel_bounds_b.bmin * static_cast<double>(sgside_a);
    rel_bounds_b.bmax = rel_bounds_b.bmax * static_cast<double>(sgside_a);
    IJK rel_cell_a = cell_a - cell_b;
    Vec3d cell_a_relative_pos = rel_cell_a * static_cast<double>(sgside_b);

    apply_grid_cell_nearby( IJK{sgside_a,sgside_a,sgside_a} , rel_bounds_b , max_dist*sgside_a ,
    [sgside_b,sgside_a,max_dist,cell_a_relative_pos,func]( IJK sg_cell_a )
    {
        Vec3d sgcell_a_relative_low = ( sg_cell_a * static_cast<double>(sgside_b) ) / static_cast<double>(sgside_a) ;
        Vec3d sgcell_a_relative_hi = ( (sg_cell_a+1) * static_cast<double>(sgside_b) ) / static_cast<double>(sgside_a) ;
        AABB sgcell_a_relative_bounds = { cell_a_relative_pos+sgcell_a_relative_low , cell_a_relative_pos+sgcell_a_relative_hi };
        double relative_dist = max_dist * sgside_b;

        apply_grid_cell_nearby( IJK{sgside_b,sgside_b,sgside_b} , sgcell_a_relative_bounds, relative_dist,
        [sg_cell_a,func]( IJK sg_cell_b )
        {
          func( sg_cell_a, sg_cell_b );
        });

    });
  }

}


