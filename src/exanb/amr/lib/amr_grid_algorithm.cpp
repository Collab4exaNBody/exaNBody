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
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>
#include <exanb/core/math_utils.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/geometry.h>
#include <exanb/core/log.h>
#include <exanb/core/operator.h>

#include <vector>
#include <set>

namespace exanb
{
  // *************************************************************************************
  // ******************** Z-Order algorithms *********************************************
  // *************************************************************************************

  void build_z_curve_order( unsigned int side, std::vector<unsigned int>& indices )
  {
    static constexpr unsigned int UNDEF_IDX = std::numeric_limits<unsigned int>::max();
    size_t n = side*side*side;
    indices.assign( n , UNDEF_IDX );
    if(n==0) { return; }
    
    unsigned int side_npo2 = 1;
    unsigned int sidelog2 = 0;
    while(side_npo2<side) { side_npo2*=2; ++sidelog2; }    
    size_t n_npo2 = side_npo2*side_npo2*side_npo2;
    
    // ldbg << "side_npo2=" << side_npo2 << ", sidelog2="<<sidelog2<<", n_npo2="<<n_npo2<<std::endl;
    
    unsigned int counter = 0;
    for(size_t c=0;c<n_npo2;c++)
    {
      unsigned int i = 0;
      unsigned int j = 0;
      unsigned int k = 0;
      size_t t = c;
      for(unsigned int b=0;b<sidelog2;b++)
      {
        i |= ( t & size_t(1) ) << b; t = t >> 1;
        j |= ( t & size_t(1) ) << b; t = t >> 1;
        k |= ( t & size_t(1) ) << b; t = t >> 1;
      }
      if( i<side && j<side && k<side )
      {
        size_t index_loc = grid_ijk_to_index( IJK{side,side,side} , IJK{i,j,k} );
        assert( indices[ index_loc ] == UNDEF_IDX );
        indices[ index_loc ] = counter++;
      }
    }

#   ifndef NDEBUG
    assert( counter == n );
    std::set<unsigned int> uind;
    for(auto x:indices)
    {
      assert( x != std::numeric_limits<unsigned int>::max() );
      assert( x>=0 && x<n );
      assert( uind.find(x) == uind.end() );
      uind.insert(x);
    }
    assert( uind.size() == n );
#   endif
  }

  static std::vector< std::vector<unsigned int> > inline build_z_curve_order_set( unsigned int side_max )
  {
    std::vector< std::vector<unsigned int> > indices_set( side_max+1 );
    for( unsigned int side=0 ; side<=side_max ; side++ )
    {
      build_z_curve_order( side , indices_set[side] );
    }
    return indices_set;
  }

  const std::vector< std::vector<unsigned int> > g_z_curve_order = build_z_curve_order_set( MAX_SUB_GRID_RESOLUTION );


  // *************************************************************************************
  // ******************** sub grid cell pairs ********************************************
  // *************************************************************************************

  void max_distance_sub_cell_pairs( OperatorDebugLogFilter & ldbg, const AmrGrid& amr, double cell_size, double max_dist, AmrSubCellPairCache& dscp_cache )
  {
    const auto& sub_grid_start = amr.sub_grid_start();
    const auto& sub_grid_cells = amr.sub_grid_cells();
    const size_t n_cells = sub_grid_start.size() - 1;
    const double max_dist2 = max_dist * max_dist;
    int max_res = 0;
    int min_res = 65535;
    size_t total_sub_cells = 0;
    size_t total_particles = 0;

#   pragma omp parallel for reduction(max:max_res) reduction(min:min_res) reduction(+:total_particles,total_sub_cells)
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      ssize_t sgsize = sub_grid_start[cell_i+1] - sub_grid_start[cell_i];        
      ssize_t n_sub_cells = sgsize + 1;
      int sgside = icbrt64(n_sub_cells);
      max_res = std::max( max_res , sgside );
      min_res = std::min( min_res , sgside );

      // debug only        
      size_t n_particles = 0;
      if( sgsize > 0 ) { n_particles = sub_grid_cells[sub_grid_start[cell_i]+sgsize-1]; }
      total_particles += n_particles;
      total_sub_cells += sgside*sgside*sgside -1;
    }

    if( static_cast<size_t>(max_res) <= dscp_cache.m_max_res && cell_size==dscp_cache.m_cell_size && max_dist==dscp_cache.m_max_dist )
    {
      ldbg << "sub grid cell pair cache up to date" << std::endl;
      return ;
    }

    assert( max_res < (1<<5) );

    int n_cell_layers = static_cast<int>( std::ceil( max_dist / cell_size ) );
    int n_nbh_cells = (n_cell_layers+1)*(n_cell_layers+1)*(n_cell_layers+1);

    ldbg <<"cell_size="<<cell_size<<", max_dist="<<max_dist<<", max res="<<max_res<<", min res="<<min_res
         <<", avg dens="<<total_particles/static_cast<double>(total_sub_cells)<<", layers="<<n_cell_layers<<", nbh_cells="<<n_nbh_cells<<std::endl;
        
    dscp_cache.m_max_res = max_res;
    dscp_cache.m_cell_size = cell_size;
    dscp_cache.m_max_dist = max_dist;
    
    auto& dscp = dscp_cache.m_sub_cell_pairs;
    dscp.clear();
    
    size_t total_full_pairs = 0;
    size_t total_partial_pairs = 0;
    size_t n_resolution_pairs = unique_pair_count(max_res+1);
    n_resolution_pairs = 0;
    
    for(int res_b=1;res_b<=max_res;res_b++)
    {
      for(int res_a=1;res_a<=res_b;res_a++)
      {
        const double sub_cell_size_a = cell_size / res_a;
        const double sub_cell_size_b = cell_size / res_b;

#       ifndef NDEBUG
        size_t res_pair_id = unique_pair_id( res_a-1 , res_b-1 );
        assert( res_pair_id == n_resolution_pairs );
#       endif
        ++ n_resolution_pairs;

        for(int cell_b_k=0;cell_b_k<=n_cell_layers;cell_b_k++)
        for(int cell_b_j=0;cell_b_j<=n_cell_layers;cell_b_j++)
        for(int cell_b_i=0;cell_b_i<=n_cell_layers;cell_b_i++)
        {
          auto& cp = dscp.emplace_back();
          
          for(int ka=0;ka<res_a;ka++)
          for(int ja=0;ja<res_a;ja++)
          for(int ia=0;ia<res_a;ia++)
          for(int kb=0;kb<res_b;kb++)
          for(int jb=0;jb<res_b;jb++)
          for(int ib=0;ib<res_b;ib++)
          {
            const AABB sc_a = {
              { ia    *sub_cell_size_a , ja    *sub_cell_size_a, ka    *sub_cell_size_a } ,
              { (ia+1)*sub_cell_size_a , (ja+1)*sub_cell_size_a, (ka+1)*sub_cell_size_a } };
              
            const AABB sc_b = {
              { cell_b_i*cell_size +  ib   *sub_cell_size_b , cell_b_j*cell_size +  jb   *sub_cell_size_b, cell_b_k*cell_size +  kb   *sub_cell_size_b } ,
              { cell_b_i*cell_size + (ib+1)*sub_cell_size_b , cell_b_j*cell_size + (jb+1)*sub_cell_size_b, cell_b_k*cell_size + (kb+1)*sub_cell_size_b } };
              
            const double mind2 = min_distance2_between( sc_a , sc_b );
            //const double maxd2 = max_distance2_between( sc_a , sc_b );

            const uint16_t ax = (ka<<10) | (ja<<5) | ia ;
            const uint16_t bx = (kb<<10) | (jb<<5) | ib ;

            /*if( maxd2 <= max_dist2 && !all_partial )
            {
              cp.m_full_a.push_back(ax);
              cp.m_full_b.push_back(bx);
              ++ total_full_pairs;
            }
            else*/ if( mind2 <= max_dist2 )
            {
              //cp.m_pair_a.push_back(ax);
              //cp.m_pair_b.push_back(bx);
              cp.m_pair_ab.push_back(ax);
              cp.m_pair_ab.push_back(bx);
              ++ total_partial_pairs;
            }
          }
          //cp.m_pair_a.shrink_to_fit();
          //cp.m_pair_b.shrink_to_fit();
          cp.m_pair_ab.shrink_to_fit();
        }
      }
    }
    assert( dscp.size() == n_resolution_pairs*n_nbh_cells );
    ldbg <<total_full_pairs<< " full, "<<total_partial_pairs<<" partial"<<std::endl;
  }
  
  
}

