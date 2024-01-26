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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <utility> // for std::pair

#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/grid_apply_particle_pair_nd.h>

#include <exanb/core/algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/grid.h>
#include <exanb/amr/amr_grid.h>

// uncomment the following line to enable profiling messages
//#define PARALLEL_BUILD_BONDS_PROFILING 1

#ifdef PARALLEL_BUILD_BONDS_PROFILING
#include <chrono>
#include <iostream>
#endif

namespace exanb
{
  

  /*
   * populate GridParticleNeighbors stores neighbors (pairs of particles) for each cell C, such that :
   * for each particle Pa in C, we have a list of neighbors Bi that connect Pa to Pb such that Pb is "after" Pa.
   * this "after" order is given by the lexicographical order (cell,particle)
   * 
   * must be called from a sequential region.
   */
  template<typename GridT, bool EncodeParticleType = true>
  static inline void parallel_build_neighbors_impl (GridT& grid, AmrGrid& amr, GridParticleNeighbors & pb, double max_dist)
  {
    using std::vector;
    using std::pair;

    assert( ! omp_in_parallel() );

    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    // wil contain encoded second members of neighbor pairs
    pb.resize( n_cells );

#   ifdef PARALLEL_BUILD_BONDS_PROFILING
    double thread_min_time = 1.e40;
    double thread_max_time = 0.;
#   endif

#   pragma omp parallel
    {
#     ifdef PARALLEL_BUILD_BONDS_PROFILING
      auto T0 = std::chrono::high_resolution_clock::now();
#     endif
    
      ssize_t current_cell_a = -1;
      vector<uint32_t> particle_neighbor_count;
      particle_neighbor_count.reserve(256);
      vector< pair<uint64_t,uint64_t> > neighbors;
      neighbors.reserve(32768);

      // WARNING!
      // we need to initialize nbh_start for all cells first, because some cells may have no particles, or particles not neighbored to any other,
      // such that we'll never traverse it in omp_grid_apply_particle_pair_nd

      // init particle types      
#     pragma omp for schedule(static)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        size_t n_particles = cells[cell_i].size();
        pb[cell_i].nbh_start.assign( n_particles , 0 );
        pb[cell_i].neighbors.clear();
      }

//      const uint64_t* __restrict__ id_b_ptr = nullptr;

      omp_grid_apply_particle_pair_nd( grid, amr, max_dist,
      // add particle pair operator
      [cells,&pb,&current_cell_a,&particle_neighbor_count,&neighbors/*,&id_b_ptr*/](ssize_t cell_a, size_t p_a, size_t cell_b, size_t p_b)
      {
        // currently processed cell changed
        if( cell_a != current_cell_a )
        {
          // end previous cell
          if( current_cell_a != -1 )
          {
            // aggregate neighbors starting at particles in previous cell          
            pb[current_cell_a].nbh_start.assign( particle_neighbor_count.begin() , particle_neighbor_count.end() );
            exanb::exclusive_prefix_sum( pb[current_cell_a].nbh_start.data() , pb[current_cell_a].nbh_start.size() );
            pb[current_cell_a].neighbors.resize( neighbors.size() );
            for( auto bnd : neighbors ) { pb[current_cell_a].neighbors[ pb[current_cell_a].nbh_start[bnd.first] ++ ] = bnd.second; }
            // at this point pb[current_cell_a].nbh_start is now an inclusive scan sum of particle neighbor counts, starting with particle count of first particle in cell
#           ifndef NDEBUG
            for(size_t i=0;i<particle_neighbor_count.size();i++)
            {
              ssize_t start=0;
              if(i>0) { start = pb[current_cell_a].nbh_start[i-1]; }
              ssize_t end = pb[current_cell_a].nbh_start[i];
              assert( (end-start) == particle_neighbor_count[i] );
            }
#           endif
            assert( pb[current_cell_a].nbh_start.size() > 0 );
            assert( pb[current_cell_a].nbh_start.back() == pb[current_cell_a].neighbors.size() );
            assert( pb[current_cell_a].nbh_start.size() == cells[current_cell_a].size() );
          }
          
          // start a new cell
          current_cell_a = cell_a;
          particle_neighbor_count.assign( cells[current_cell_a].size() , 0 );
          neighbors.clear();
        }
        
//        id_ptr
        
        uint64_t p_id_b = exanb::encode_cell_particle(cell_b,p_b);
        neighbors.push_back( pair<uint64_t,uint64_t>(p_a,p_id_b) );
        ++ particle_neighbor_count[p_a];
      }
      ,
      std::false_type() // no wait
      );

      // end last traversed cell
      if( current_cell_a != -1 )
      {
        // aggregate neighbors starting at particles in previous cell          
        pb[current_cell_a].nbh_start.assign( particle_neighbor_count.begin() , particle_neighbor_count.end() );
        exanb::exclusive_prefix_sum( pb[current_cell_a].nbh_start.data() , pb[current_cell_a].nbh_start.size() );
        pb[current_cell_a].neighbors.resize( neighbors.size() );
        for( auto bnd : neighbors ) { pb[current_cell_a].neighbors[ pb[current_cell_a].nbh_start[bnd.first] ++ ] = bnd.second; }
        // at this point pb[current_cell_a].nbh_start is now an inclusive scan sum of particle neighbor counts, starting with particle count of first particle in cell
#       ifndef NDEBUG
        for(size_t i=0;i<particle_neighbor_count.size();i++)
        {
          ssize_t start=0;
          if(i>0) { start = pb[current_cell_a].nbh_start[i-1]; }
          ssize_t end = pb[current_cell_a].nbh_start[i];
          assert( (end-start) == particle_neighbor_count[i] );
        }
#       endif
        assert( pb[current_cell_a].nbh_start.size() > 0 );
        assert( pb[current_cell_a].nbh_start.back() == pb[current_cell_a].neighbors.size() );
        assert( pb[current_cell_a].nbh_start.size() == cells[current_cell_a].size() );
      }

#     ifdef PARALLEL_BUILD_BONDS_PROFILING
      double thread_time = (std::chrono::high_resolution_clock::now() - T0).count() / 1000000.0;
#     pragma omp critical
      {
        thread_max_time = std::max( thread_max_time , thread_time );
        thread_min_time = std::min( thread_min_time , thread_time );
      }
#     endif

    } // omp parallel

#   ifdef PARALLEL_BUILD_BONDS_PROFILING
    std::cout<<"parallel_build_neighbors thread time (min/max) = "<<thread_min_time<<" / "<<thread_max_time<<std::endl;
#   endif
  }

}

