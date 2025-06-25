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

#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>
#include <onika/math/math_utils.h>
#include <exanb/core/algorithm.h>
#include <onika/log.h>
#include <onika/scg/operator.h>

#ifndef NDEBUG
#include <onika/math/basic_types_stream.h>
#endif

#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

// uncomment the following to scramble particles order before AMR re-ordering (for perf tests only)
// #define XNB_AMR_RANDOMIZE_PARTICLES 1

namespace exanb
{
  extern const std::vector< std::vector<unsigned int> > g_z_curve_order;
  extern void build_z_curve_order( unsigned int side, std::vector<unsigned int>& indices );

  // maximum sub grid resolution is 16x16x16
  static constexpr size_t MAX_SUB_GRID_RESOLUTION = 16;
  static constexpr bool SUB_GRID_RESTRICT_TO_PO2 = false;

  static inline unsigned long upper_power_of_two(unsigned long v)
  {
      -- v;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      ++ v;
      return v;
  }

  // computes the dimensions of a cubic grid so that its density per cell would be the nearest possible to avg_density if particles are uniformly distributed
  // guarantees that :
  //  if n==0, returned grid size is 0
  //  if n>0, returned grid size is >=1
  //  s^3 <= n <= (s+1)^3
  static inline size_t sub_grid_size(size_t n, double avg_density=1.0)
  {
    if( n == 0 ) { return 0; }
    double nbiased = n / avg_density;
    double side = std::cbrt(nbiased);
    if( side < 2.0 ) { return 1; }
    size_t iside = static_cast<size_t>( std::floor( side ) );
    if( SUB_GRID_RESTRICT_TO_PO2 )
    {
      iside = upper_power_of_two(iside);
    }
    return std::min( iside , MAX_SUB_GRID_RESOLUTION );
  }

  template<bool EnableZOrder=false>
  ONIKA_HOST_DEVICE_FUNC inline unsigned int sg_cell_index( ssize_t sgside, IJK loc, std::integral_constant<bool,EnableZOrder> = {} )
  {
    if constexpr ( EnableZOrder )
    {
      unsigned int table_index = grid_ijk_to_index( IJK{sgside,sgside,sgside} , loc );
      assert( table_index < g_z_curve_order[sgside].size() );
      return g_z_curve_order[sgside][table_index];
    }
    if constexpr ( ! EnableZOrder )
    {
      return grid_ijk_to_index( IJK{sgside,sgside,sgside} , loc );
    }
    return 0;
  }

  template<unsigned int SUBRES>
  inline ssize_t sg_cell_subres_index(ssize_t ssgi, ssize_t ssgj, ssize_t ssgk, std::integral_constant<unsigned int,SUBRES> )
  {
    ssize_t zc = 0;
    if constexpr ( SUBRES == 4 )
    { // 3D 2-bits Z-Order
      zc = ( (ssgk&2)<<4 ) | ( (ssgj&2)<<3 ) | ( (ssgi&2)<<2 ) | ( (ssgk&1)<<2 ) | ( (ssgj&1)<<1 ) | (ssgi&1) /*<<0*/ ;
    }
    if constexpr ( SUBRES != 4 )
    {
      zc = ssgk*SUBRES*SUBRES + ssgj*SUBRES + ssgi;
    }
    assert( zc>=0 && zc<(SUBRES*SUBRES*SUBRES) );
    return zc;
  }

  template<class LDBG, class GridT, bool EnableZOrder=false, unsigned int SUBRES=1>
  inline void project_particles_in_sub_grids( LDBG& ldbg, GridT& grid , AmrGrid& amr , std::integral_constant<bool,EnableZOrder> enable_z_order = {} , std::integral_constant<unsigned int,SUBRES> subres = {} )
  {    
    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();
    IJK dims = grid.dimension();

    auto& sub_grid_start = amr.sub_grid_start();
    auto& sub_grid_cells = amr.sub_grid_cells();

    // if sub grids where not updated, defaults to no sub grids (each has 0 size)
    if( sub_grid_start.size() != (n_cells+1) )
    {
      amr.clear_sub_grids( grid.number_of_cells() );
      return;
    }

    assert( sub_grid_start.size() >= n_cells+1 );

#   pragma omp parallel
    {      
      std::vector<int32_t> sub_grid_index;
      std::vector<int32_t> sub_grid_source;
      std::vector<uint32_t> sub_grid_cells_subres_storage;

#     pragma omp for schedule(dynamic)
      for(size_t cell_i=0; cell_i<n_cells; cell_i++)
      {
        ssize_t n_particles = cells[cell_i].size();
        ssize_t sgstart = sub_grid_start[cell_i];
        ssize_t sgsize = sub_grid_start[cell_i+1] - sub_grid_start[cell_i];
        IJK cell_loc = grid_index_to_ijk(dims,cell_i);

        if( sgsize > 0 || SUBRES > 1 )
        {
          // Note: sgsize is 1 less than the number of sub grid cells 
          ssize_t n_sub_cells = sgsize+1;
          ssize_t sgside = std::floor( std::cbrt( n_sub_cells ) + 0.5 );
          assert( (sgside*sgside*sgside) == n_sub_cells );
          assert( sgside == icbrt64(n_sub_cells) );

          ssize_t n_sub_cells_sr = n_sub_cells*SUBRES*SUBRES*SUBRES;
          ssize_t sgsize_sr = n_sub_cells_sr - 1;
                    
          // resize temporary arrays if necessary. avoid reallocating it as much as possible, so it will only grow
          sub_grid_index.resize(n_particles);
          sub_grid_source.resize(n_particles);

          if constexpr ( SUBRES > 1 ) { sub_grid_cells_subres_storage.resize( sgsize_sr ); }
          if constexpr ( SUBRES == 1 ) { assert( sgsize == sgsize_sr ); }
          uint32_t * sub_grid_cells_tmp = ( SUBRES == 1 ) ? (sub_grid_cells.data() + sgstart) : sub_grid_cells_subres_storage.data();
          
          // sub_grid_cells [sgstart,sgstart+sgsize[ contains count (and then offsets) of particles in sub grid cells
          for(ssize_t sgindex=0; sgindex<sgsize_sr; sgindex++)
          {
            sub_grid_cells_tmp[sgindex] = 0;
          }

#         ifdef XNB_AMR_RANDOMIZE_PARTICLES
          // scramble particles in cell (for performance testing purposes only)
          for(ssize_t p=0;p<n_particles;p++)
          {
            auto t = cells[cell_i][p];
            Vec3d r = grid.particle_pcoord( cell_loc, Vec3d{t[field::rx],t[field::ry],t[field::rz]} );
            ssize_t sgi = static_cast<ssize_t>( r.x * n_particles ) + 1;
            ssize_t sgj = static_cast<ssize_t>( r.y * n_particles ) + 1;
            ssize_t sgk = static_cast<ssize_t>( r.z * n_particles ) + 1;
            ssize_t i = ( sgi*sgj*sgk ) % n_particles;
            if( i != p ) cells[cell_i].swap( p , i );
          }
#         endif

          // count number of particles in each sub grid cell
#         ifndef NDEBUG
          ssize_t last_sub_cell_count = 0;
#         endif
          for(int p=0;p<n_particles;p++)
          {
            auto t = cells[cell_i][p];
            Vec3d r = grid.particle_pcoord( cell_loc, Vec3d{t[field::rx],t[field::ry],t[field::rz]} );
            ssize_t sgi = ::exanb::clamp( static_cast<ssize_t>( r.x * sgside ) , 0l , sgside-1 );
            ssize_t sgj = ::exanb::clamp( static_cast<ssize_t>( r.y * sgside ) , 0l , sgside-1 );
            ssize_t sgk = ::exanb::clamp( static_cast<ssize_t>( r.z * sgside ) , 0l , sgside-1 );
            ssize_t sgindex = sg_cell_index( sgside , IJK{sgi, sgj, sgk} , enable_z_order ); // grid_ijk_to_index( IJK{sgside,sgside,sgside} , IJK{sgi, sgj, sgk} );
            if constexpr ( SUBRES > 1 )
            {
              ssize_t ssgi = ::exanb::clamp( static_cast<ssize_t>( r.x * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
              ssize_t ssgj = ::exanb::clamp( static_cast<ssize_t>( r.y * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
              ssize_t ssgk = ::exanb::clamp( static_cast<ssize_t>( r.z * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
              sgindex = sgindex * (SUBRES*SUBRES*SUBRES) + sg_cell_subres_index(ssgi,ssgj,ssgk,subres);
            }
            assert( sgindex>=0 && sgindex<n_sub_cells_sr );
            sub_grid_index[p] = sgindex;
            sub_grid_source[p] = -1;
            if( sgindex < sgsize_sr )
            {
              ++ sub_grid_cells_tmp[sgindex];
            }
#           ifndef NDEBUG
            else { ++ last_sub_cell_count; }
#           endif
          }
          // sub_grid_cells[sgstart .. sgstart+sgsize-1] = nb of particles in sub cells 0 .. sgsize-1 (n_sub_cells-2)

          for(ssize_t sgindex=1; sgindex<sgsize_sr; sgindex++)
          {
            sub_grid_cells_tmp[sgindex] += sub_grid_cells_tmp[sgindex-1];
          }
          // sub_grid_cells[sgstart .. sgstart+sgsize-1] = N0 , N0+N1 , N0+N1+N2 ...
          assert( sgsize_sr==0 || ( ( sub_grid_cells_tmp[sgsize_sr-1] + last_sub_cell_count ) == n_particles ) );

          // compute final index of particle w.r.t its containing sub cell
          ssize_t first_sub_cell_count = 0;
          for(int p=0;p<n_particles;p++)
          {
            ssize_t sgindex = sub_grid_index[p];
            assert( sgindex>=0 && sgindex<n_sub_cells_sr );
            if( sgindex > 0 ) { sub_grid_index[p] = sub_grid_cells_tmp[sgindex-1] ++; }
            else { sub_grid_index[p] = first_sub_cell_count++; }
            assert( sub_grid_index[p]>=0 && sub_grid_index[p]<n_particles );
          }
          // after this, count==N0 (number of particles in sub grid cell #0)
          // sub_grid_cells[sgstart .. sgstart+sgsize-1] = N0+N1 , N0+N1+N2 , N0+N1+N2+N3 ...
          assert( sgsize_sr==0 || ( sub_grid_cells_tmp[sgsize_sr-1] == n_particles ) );

          // compute reverse function of sub_grid_index : where to take item that must go to p index;
          for(int p=0;p<n_particles;p++)
          {
            ssize_t sgi = sub_grid_index[p];
            assert( sgi>=0 && sgi<n_particles );
            assert( sub_grid_source[sgi] == -1 );
            sub_grid_source[sgi] = p;
            //sub_grid_source[sub_grid_index[p]] = p;
          }

#         ifndef NDEBUG
          for(ssize_t p=0;p<n_particles;p++)
          {
            assert( sub_grid_source[ sub_grid_index[p] ] == p );
            assert( sub_grid_index[ sub_grid_source[p] ] == p );
          }
#         endif

          // sort particles accroding to their destination sub grid cell
          for(ssize_t a=0;a<n_particles;a++)
          {
            ssize_t b = sub_grid_source[a];
            assert( a <= b );
            if( a != b )
            {
              assert( b>=0 && b<n_particles );
              assert( a < b );              
              assert( sub_grid_index[b] == a );
              ssize_t a_dst = sub_grid_index[a];
              assert( a < a_dst );              
              
              cells[cell_i].swap( a, b );

              sub_grid_source[a_dst] = b;              
              sub_grid_index[b] = a_dst;
              
              sub_grid_source[a] = a;
              sub_grid_index[a] = a;
              
              assert( sub_grid_index[sub_grid_source[a_dst]] == a_dst );
              assert( sub_grid_source[sub_grid_index[a_dst]] == a_dst );
              assert( sub_grid_index[sub_grid_source[b]] == b );
              assert( sub_grid_source[sub_grid_index[b]] == b );
            }
          }

#         ifndef NDEBUG
          for(ssize_t p=0;p<n_particles;p++)
          {
            assert( sub_grid_source[p] == p );
            assert( sub_grid_index[p] == p );
          }
#         endif

          // recover content of sub_grid_cells [sgstart,sgstart+sgsize[
          ssize_t prev_count = first_sub_cell_count;
          for(ssize_t sgindex=0; sgindex<sgsize_sr; sgindex++)
          {
            ssize_t nextCount = sub_grid_cells_tmp[sgindex];
            sub_grid_cells_tmp[sgindex] = prev_count;
            prev_count = nextCount;
          }
          // after this, m_sub_grid_index contains cumulated sub grid cell nb of particles : N0, N0+N1, N0+N1+N2, ...

          if constexpr ( SUBRES > 1 )
          {
            // check that new particles ordering satisfies high resolution sub grid classification
#           ifndef NDEBUG
            for(ssize_t sgcell=0; sgcell<n_sub_cells_sr; sgcell++)
            {
              size_t beginp = 0;
              if( sgcell > 0 ) { beginp = sub_grid_cells_tmp[sgcell-1]; }
              size_t endp = n_particles;
              if( sgcell < sgsize_sr ) { endp = sub_grid_cells_tmp[sgcell]; }
              //assert( beginp >= 0 );
              assert( ssize_t(beginp) <= n_particles );
              assert( endp >= beginp );
              assert( ssize_t(endp) <= n_particles );
              for(size_t p=beginp;p<endp;p++)
              {
                auto t = cells[cell_i][p];
                Vec3d r = grid.particle_pcoord( cell_loc, Vec3d{t[field::rx],t[field::ry],t[field::rz]} );
                ssize_t sgi = ::exanb::clamp( static_cast<ssize_t>( r.x * sgside ) , 0l , sgside-1 );
                ssize_t sgj = ::exanb::clamp( static_cast<ssize_t>( r.y * sgside ) , 0l , sgside-1 );
                ssize_t sgk = ::exanb::clamp( static_cast<ssize_t>( r.z * sgside ) , 0l , sgside-1 );
                ssize_t sgindex = sg_cell_index( sgside , IJK{sgi, sgj, sgk} , enable_z_order ); // grid_ijk_to_index( IJK{sgside,sgside,sgside} , IJK{sgi, sgj, sgk} );
                if constexpr ( SUBRES > 1 )
                {
                  ssize_t ssgi = ::exanb::clamp( static_cast<ssize_t>( r.x * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
                  ssize_t ssgj = ::exanb::clamp( static_cast<ssize_t>( r.y * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
                  ssize_t ssgk = ::exanb::clamp( static_cast<ssize_t>( r.z * sgside * SUBRES ) , 0l , (sgside*SUBRES)-1 ) % SUBRES;
                  sgindex = sgindex * (SUBRES*SUBRES*SUBRES) + sg_cell_subres_index(ssgi,ssgj,ssgk,subres);
                  assert( sgindex>=0 && sgindex<n_sub_cells_sr );
                }
                assert( sgindex == sgcell );
              }
            }
#           endif

            // rebuild low resolution offset table from high resolution offset table
            for(ssize_t sgindex=0; sgindex<sgsize; sgindex++)
            {
              sub_grid_cells[sgstart+sgindex] = sub_grid_cells_tmp[(sgindex+1)*SUBRES*SUBRES*SUBRES-1];
            }
          }

          // check that particles are correctly ordered
#         ifndef NDEBUG
          for(ssize_t sgcell=0; sgcell<n_sub_cells; sgcell++)
          {
            size_t beginp = 0;
            if( sgcell > 0 ) { beginp = sub_grid_cells[sgstart+sgcell-1]; }
            size_t endp = n_particles;
            if( sgcell < sgsize ) { endp = sub_grid_cells[sgstart+sgcell]; }
            //assert( beginp >= 0 );
            assert( ssize_t(beginp) <= n_particles );
            assert( endp >= beginp );
            assert( ssize_t(endp) <= n_particles );
            for(size_t p=beginp;p<endp;p++)
            {
              auto t = cells[cell_i][p];
              Vec3d r = grid.particle_pcoord( cell_loc, Vec3d{t[field::rx],t[field::ry],t[field::rz]} );
              ssize_t sgi = ::exanb::clamp( static_cast<ssize_t>( r.x * sgside ) , 0l , sgside-1 );
              ssize_t sgj = ::exanb::clamp( static_cast<ssize_t>( r.y * sgside ) , 0l , sgside-1 );
              ssize_t sgk = ::exanb::clamp( static_cast<ssize_t>( r.z * sgside ) , 0l , sgside-1 );
              ssize_t sgindex = sg_cell_index( sgside , IJK{sgi, sgj, sgk} , enable_z_order ); // grid_ijk_to_index( IJK{sgside,sgside,sgside} , IJK{sgi, sgj, sgk} );
              assert( sgindex == sgcell );
            }              
          }
#         endif

        } // end if sgsize>0
        //else { assert( m_sub_grid_side[cell_i] == 0 ); }
        
      } // end of parallel for on cells
    
    } // end of parallel region      
  }


  // contain parallel regions, must be called from a sequential region.
  template<class LDBG, class GridT, bool EnableZOrder=false , unsigned int SUBRES=1>
  inline void rebuild_sub_grids( LDBG& ldbg, GridT& grid, AmrGrid& amr, double avg_density , std::integral_constant<bool,EnableZOrder> enable_z_order = std::false_type() , std::integral_constant<unsigned int,SUBRES> subres = {}  )
  {
    const size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();
    
    AmrGrid::SubGridStartVector & sub_grid_start = amr.sub_grid_start();
    AmrGrid::SubGridCellsVector & sub_grid_cells = amr.sub_grid_cells();

    sub_grid_cells.clear();
    sub_grid_cells.shrink_to_fit();
//    sub_grid_start.clear();
//    sub_grid_start.shrink_to_fit();
    sub_grid_start.assign( n_cells + 1 , 0 );

    //m_sub_grid_side.resize( n_cells );
    [[maybe_unused]] size_t total_grid_size = 0;
    sub_grid_start[0] = 0;

    int max_res = 0;
    int min_res = 65535;

#   pragma omp parallel for reduction(+:total_grid_size) reduction(min:min_res) reduction(max:max_res)
    for(size_t i=0;i<n_cells;i++)
    {
      int grid_side = sub_grid_size( cells[i].size() , avg_density );
      max_res = std::max( max_res , grid_side );
      min_res = std::min( min_res , grid_side );
      int grid_size = std::max( grid_side*grid_side*grid_side - 1 , 0 );
      sub_grid_start[i+1] = grid_size;
      total_grid_size += grid_size;
    }
    
    // scan sum
    // TODO: replace with parallel scan sum
    std::partial_sum( sub_grid_start.begin(), sub_grid_start.end(), sub_grid_start.begin() );
    assert( sub_grid_start[0] == 0 );
    assert( sub_grid_start[n_cells] == total_grid_size );

    sub_grid_cells.resize( sub_grid_start[n_cells] );

    ldbg << "AMR resolution : "<<min_res<<" - "<<max_res<<std::endl;
    ldbg << "data size : sub_grid_start="<<sub_grid_start.size()<<" , sub_grid_cells="<<sub_grid_cells.size()<<std::endl;
    
/*
#   ifndef NDEBUG
    for(size_t i=0;i<n_cells;i++)
    {
      ssize_t grid_side = sub_grid_size( cells[i].size() , avg_density );
      size_t grid_size = std::max( grid_side*grid_side*grid_side - 1 , static_cast<ssize_t>(0) );
      assert( ( sub_grid_start[i+1] - sub_grid_start[i] ) == grid_size );
    }      
#   endif
*/

    project_particles_in_sub_grids( ldbg, grid , amr , enable_z_order , subres );
  }    


  // accelerating structure for sub grid cell pairs

  struct AmrSubCellPairs
  {
    std::vector<uint16_t> m_pair_ab;
    //std::vector<uint16_t> m_pair_b;
  };

  struct AmrSubCellPairCache
  {
    double m_cell_size=0.;
    double m_max_dist=0.;
    size_t m_max_res=0;
    std::vector<AmrSubCellPairs> m_sub_cell_pairs;
    inline size_t cell_layers() const { return static_cast<size_t>( std::ceil( m_max_dist / m_cell_size ) ) ; }
    inline size_t nb_nbh_cells() const { size_t x = cell_layers()+1; return x*x*x; }
  };

  void max_distance_sub_cell_pairs( onika::scg::OperatorDebugLogFilter & ldbg, const AmrGrid& amr, double cell_size, double max_dist, AmrSubCellPairCache& dscp );
}

