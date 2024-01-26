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

#include <exanb/core/grid.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/parallel_grid_algorithm.h>

namespace exanb
{
  

   namespace cst
   {
    static constexpr double sqrt_3 = 1.7320508075688772; //std::sqrt(3.);
   }
   
  // *** particle neighborhood ***
  // apply a functor on every pair of particles that are closer than specified distance
  template<typename PartPairFuncT, typename GridFieldSetT >
  static inline void omp_grid_apply_particle_pair_nd(
    Grid< GridFieldSetT >& grid,
    AmrGrid& amr,
    double max_dist,
    PartPairFuncT func,
    bool omp_wait_opt=true )
  {
    using GridT = Grid< GridFieldSetT >;
    using CellT = typename GridT::CellParticles;
    
    CellT* grid_cells = grid.cells();
    const size_t* sub_grid_start = amr.sub_grid_start().data();
    const uint32_t* sub_grid_cells = amr.sub_grid_cells().data();
    double cell_size = grid.cell_size();
    IJK grid_dim = grid.dimension();
    
  # ifndef NDEBUG
    size_t n_cells = grid.number_of_cells();
    assert( amr.sub_grid_start().size() == (n_cells+1) );
  # endif
    
    double cell_max_dist = max_dist / cell_size;
    double max_dist2 = max_dist*max_dist;

    // for all grid cell pairs, such that minimal distance between those two cells is less or equal to max_dist
    omp_apply_grid_block_cell_pair( grid_dim, IJK{0,0,0}, grid_dim, cell_max_dist,

    // cell pair operator. here we'll look for particle pairs between two grid cells
    [grid_cells,sub_grid_start,sub_grid_cells,grid_dim,/*max_dist,*/max_dist2,cell_max_dist,func](IJK cell_a, IJK cell_b)
    {
      size_t cell_a_i = grid_ijk_to_index(grid_dim, cell_a);
      size_t n_particles_a = grid_cells[cell_a_i].size();
      size_t sgstart_a = sub_grid_start[cell_a_i];
      ssize_t sgsize_a = sub_grid_start[cell_a_i+1] - sub_grid_start[cell_a_i];
      ssize_t n_sub_cells_a = sgsize_a+1;
      ssize_t sgside_a = /* static_cast<ssize_t>(m_sub_grid_side[cell_a_i]) + 1; */ std::cbrt( n_sub_cells_a );
      const double* __restrict__ rx_a = grid_cells[cell_a_i][field::rx];
      const double* __restrict__ ry_a = grid_cells[cell_a_i][field::ry];
      const double* __restrict__ rz_a = grid_cells[cell_a_i][field::rz];
      
      size_t cell_b_i = grid_ijk_to_index(grid_dim, cell_b);
      size_t n_particles_b = grid_cells[cell_b_i].size();
      size_t sgstart_b = sub_grid_start[cell_b_i];
      ssize_t sgsize_b = sub_grid_start[cell_b_i+1] - sub_grid_start[cell_b_i];
      ssize_t n_sub_cells_b = sgsize_b+1;
      ssize_t sgside_b = /* static_cast<ssize_t>(m_sub_grid_side[cell_b_i]) + 1; */ std::cbrt( n_sub_cells_b );
      const double* __restrict__ rx_b = grid_cells[cell_b_i][field::rx];
      const double* __restrict__ ry_b = grid_cells[cell_b_i][field::ry];
      const double* __restrict__ rz_b = grid_cells[cell_b_i][field::rz];

      // variant 2
      apply_inter_cell_sub_grid_cell_pair(cell_a, sgside_a, cell_b, sgside_b, cell_max_dist,
      [sub_grid_cells,max_dist2,cell_a_i,cell_b_i,sgside_a,sgside_b,n_particles_a,n_particles_b,sgstart_a,sgstart_b,sgsize_a,sgsize_b, rx_a,ry_a,rz_a, rx_b,ry_b,rz_b, func](IJK sgcell_a, IJK sgcell_b)
      {
        ssize_t sgindex_a = grid_ijk_to_index( IJK{sgside_a,sgside_a,sgside_a} , sgcell_a );
        ssize_t p_start_a = 0;
        ssize_t p_end_a = n_particles_a;
        if( sgindex_a > 0 ) { p_start_a = sub_grid_cells[sgstart_a+sgindex_a-1]; }
        if( sgindex_a < sgsize_a ) { p_end_a = sub_grid_cells[sgstart_a+sgindex_a]; }

        ssize_t sgindex_b = grid_ijk_to_index( IJK{sgside_b,sgside_b,sgside_b} , sgcell_b );
        ssize_t p_start_b = 0;
        ssize_t p_end_b = n_particles_b;
        if( sgindex_b > 0 ) { p_start_b = sub_grid_cells[sgstart_b+sgindex_b-1]; }
        if( sgindex_b < sgsize_b ) { p_end_b = sub_grid_cells[sgstart_b+sgindex_b]; }

        for(ssize_t pa_i=p_start_a;pa_i<p_end_a;pa_i++)
        {
          Vec3d pa = { rx_a[pa_i], ry_a[pa_i], rz_a[pa_i] };
          for(ssize_t pb_i=p_start_b;pb_i<p_end_b;pb_i++)
          {
            Vec3d pb = { rx_b[pb_i], ry_b[pb_i], rz_b[pb_i] };
            double d2 = distance2( pa , pb );
            if( d2 <= max_dist2 )
            {
              func(cell_a_i,pa_i,cell_b_i,pb_i);
            }
          }
        }        
      });      
    }
    ,
    
    // cell startup operator. here we'll search for particle pairs inside a cell (grid cell).
    [grid_dim,grid_cells,sub_grid_start,sub_grid_cells,max_dist2,cell_max_dist,func](IJK cell)
    {
      //IJK cell = grid_index_to_ijk(grid_dim,cell_i);
      size_t cell_i = grid_ijk_to_index(grid_dim, cell);
      size_t n_particles = grid_cells[cell_i].size();
      size_t sgstart = sub_grid_start[cell_i];
      ssize_t sgsize = sub_grid_start[cell_i+1] - sub_grid_start[cell_i];
      ssize_t n_sub_cells = sgsize+1;
      ssize_t sgside = /*static_cast<ssize_t>(m_sub_grid_side[cell_i]) + 1;*/ std::cbrt( n_sub_cells );
      assert( (sgside*sgside*sgside) == n_sub_cells );

      const double sg_cell_max_dist = cell_max_dist * sgside;
      const bool check_intra_sg_cell_dist = sg_cell_max_dist < cst::sqrt_3;
      double* rx = grid_cells[cell_i][field::rx];
      double* ry = grid_cells[cell_i][field::ry];
      double* rz = grid_cells[cell_i][field::rz];
      
      // inside a cell, search for pairs in each sub grid cells, and then between sub grid cell pair.
      apply_grid_block_cell_pair_fp( IJK{sgside,sgside,sgside}, IJK{0,0,0}, IJK{sgside,sgside,sgside}, sg_cell_max_dist,
      // pair of sub grid cells totally enclosed in a sphere of diameter <= max_dist
      [sub_grid_cells,sgstart,n_particles,cell_i,sgside,sgsize, func](IJK sub_cell_a, IJK sub_cell_b)
      {
        ssize_t sgindex_a = grid_ijk_to_index( IJK{sgside,sgside,sgside} , sub_cell_a ) ;
        size_t p_start_a = 0;
        size_t p_end_a = n_particles;
        if( sgindex_a > 0 ) { p_start_a = sub_grid_cells[sgstart+sgindex_a-1]; }
        if( sgindex_a < sgsize ) { p_end_a = sub_grid_cells[sgstart+sgindex_a]; }

        ssize_t sgindex_b = grid_ijk_to_index( IJK{sgside,sgside,sgside} , sub_cell_b ) ;
        size_t p_start_b = 0;
        size_t p_end_b = n_particles;
        if( sgindex_b > 0 ) { p_start_b = sub_grid_cells[sgstart+sgindex_b-1]; }
        if( sgindex_b < sgsize ) { p_end_b = sub_grid_cells[sgstart+sgindex_b]; }
        
        for(size_t pa_i=p_start_a; pa_i<p_end_a; pa_i++)
        {
          for(size_t pb_i=p_start_b; pb_i<p_end_b; pb_i++)
          {
            func(cell_i,pa_i,cell_i,pb_i);
          }
        }
      }
      ,
      // pair of sub grid cells partially enclosed in a sphere of diameter <= max_dist
      [sub_grid_cells,sgstart,n_particles,cell_i,sgside,sgsize,max_dist2, rx,ry,rz, func](IJK sub_cell_a, IJK sub_cell_b)
      {
        ssize_t sgindex_a = grid_ijk_to_index( IJK{sgside,sgside,sgside} , sub_cell_a ) ;
        size_t p_start_a = 0;
        size_t p_end_a = n_particles;
        if( sgindex_a > 0 ) { p_start_a = sub_grid_cells[sgstart+sgindex_a-1]; }
        if( sgindex_a < sgsize ) { p_end_a = sub_grid_cells[sgstart+sgindex_a]; }

        ssize_t sgindex_b = grid_ijk_to_index( IJK{sgside,sgside,sgside} , sub_cell_b ) ;
        size_t p_start_b = 0;
        size_t p_end_b = n_particles;
        if( sgindex_b > 0 ) { p_start_b = sub_grid_cells[sgstart+sgindex_b-1]; }
        if( sgindex_b < sgsize ) { p_end_b = sub_grid_cells[sgstart+sgindex_b]; }
        
        for(size_t pa_i=p_start_a; pa_i<p_end_a; pa_i++)
        {
          Vec3d pa = { rx[pa_i], ry[pa_i], rz[pa_i] }; //particle_position(cell_i,pa_i);
          for(size_t pb_i=p_start_b; pb_i<p_end_b; pb_i++)
          {
            Vec3d pb = { rx[pb_i], ry[pb_i], rz[pb_i] }; //particle_position(cell_i,pb_i);
            double d2 = distance2( pa , pb );
            if( d2 <= max_dist2 ) { func(cell_i,pa_i,cell_i,pb_i); }
          }
        }
      }
      ,
      // for pairs inside a sub grid cell
      [sub_grid_cells,cell_i,n_particles,sgstart,sgsize,sgside,max_dist2, rx,ry,rz, check_intra_sg_cell_dist, func](IJK sub_cell)
      {
        ssize_t sgindex = grid_ijk_to_index( IJK{sgside,sgside,sgside} , sub_cell ) ;
        size_t p_start = 0;
        size_t p_end = n_particles;
        if( sgindex > 0 ) { p_start = sub_grid_cells[sgstart+sgindex-1]; }
        if( sgindex < sgsize ) { p_end = sub_grid_cells[sgstart+sgindex]; }
        if( check_intra_sg_cell_dist )
        {
          for(size_t p_i=p_start; p_i<p_end; p_i++)
          {
            Vec3d pa = { rx[p_i], ry[p_i], rz[p_i] }; //particle_position(cell_i,p_i);
            for(size_t p_j=p_i+1; p_j<p_end; p_j++)
            {
              Vec3d pb = { rx[p_j], ry[p_j], rz[p_j] }; //particle_position(cell_i,p_j);
              double d2 = distance2( pa , pb );
              if( d2 <= max_dist2 ) { func(cell_i,p_i,cell_i,p_j); }
            }              
          }
        }
        else
        {
          for(size_t p_i=p_start; p_i<p_end; p_i++)
          {
            for(size_t p_j=p_i+1; p_j<p_end; p_j++)
            {
              func(cell_i,p_i,cell_i,p_j);
            }
          }
        }
      });
        
    }
    , null_cell_operator
    , omp_wait_opt // omp wait option passed to apply_grid_cell_pair
    ); // end of omp_apply_grid_block_cell_pair
    
  }
    
} // namespace exanb


