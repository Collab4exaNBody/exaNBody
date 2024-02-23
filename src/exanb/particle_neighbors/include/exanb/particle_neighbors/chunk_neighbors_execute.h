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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>

#include <onika/cuda/cuda_context.h>
#include <onika/memory/allocator.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_scratch.h>
#include <exanb/particle_neighbors/chunk_neighbors_host_write_accessor.h>
#include <exanb/particle_neighbors/neighbor_filter_func.h>

namespace exanb
{
    template<class LDBG, class GridT, class ChunkSizeT, class ChunkSizeLog2T, class OptionalXFormT, bool EnableZOrder, class NeighborFilterFuncT = DefaultNeighborFilterFunc >
    inline void chunk_neighbors_execute(
       LDBG& ldbg,
		   GridChunkNeighbors& chunk_neighbors,
		   const GridT& grid,
		   const AmrGrid & amr,
		   const AmrSubCellPairCache& amr_grid_pairs, 
 		   const ChunkNeighborsConfig& config,
		   ChunkNeighborsScratchStorage& chunk_neighbors_scratch,
		   ChunkSizeT cs, 
		   ChunkSizeLog2T cs_log2,
		   const double nbh_dist_lab,
		   const OptionalXFormT& xform,
		   const bool gpu_enabled,
		   std::integral_constant<bool,EnableZOrder> enable_z_order,
		   NeighborFilterFuncT nbh_filter = {} )
    {
      //using PointerTuple = onika::soatl::FieldPointerTuple< GridT::CellParticles::Alignment , GridT::CellParticles::ChunkSize , field::_rx, field::_ry, field::_rz >;

      if( static_cast<size_t>(config.chunk_size) > GRID_CHUNK_NBH_MAX_CHUNK_SIZE )
      {
        lerr << "chunk_size ("<< (config.chunk_size) <<") beyond the limit of "<<GRID_CHUNK_NBH_MAX_CHUNK_SIZE<<std::endl;
        std::abort();
      }

      chunk_neighbors.m_alloc.set_gpu_addressable_allocation( gpu_enabled );

      bool build_particle_offset = config.build_particle_offset;
      if( gpu_enabled && !build_particle_offset && config.chunk_size>1) // specialization for chunk_size=1 now suports list traversal without offset table
      {
        ldbg << "INFO: force build_particle_offset to true to ensure Cuda compatibility" << std::endl;
        build_particle_offset = true;
      }
      if( ! build_particle_offset )
      {
        ldbg << "INFO: no particle offset" << std::endl;
      }

      auto cells = grid.cells();
      IJK dims = grid.dimension();
//      ssize_t gl = grid->ghost_layers();

      const size_t* sub_grid_start = amr.sub_grid_start().data();
      const uint32_t* sub_grid_cells = amr.sub_grid_cells().data();

      //const double cell_size = grid->cell_size();
      const double max_dist = nbh_dist_lab;
      const double max_dist2 = max_dist*max_dist;
      
      const unsigned int loc_max_gap = amr_grid_pairs.cell_layers();
      const unsigned int nbh_cell_side = loc_max_gap+1;
      const unsigned int n_nbh_cell = amr_grid_pairs.nb_nbh_cells();
      assert( nbh_cell_side*nbh_cell_side*nbh_cell_side == n_nbh_cell );
            
      const size_t n_cells = grid.number_of_cells();
      
      chunk_neighbors.clear();
      chunk_neighbors.set_number_of_cells( n_cells );
      chunk_neighbors.set_chunk_size( cs );
      chunk_neighbors.realloc_stream_pool( config.stream_prealloc_factor );

      ldbg << "cell max gap = "<<loc_max_gap<<", cslog2="<<cs_log2<<", n_nbh_cell="<<n_nbh_cell<<", pre-alloc="<<chunk_neighbors.m_fixed_stream_pool.m_capacity <<std::endl;
      
      unsigned int max_threads = omp_get_max_threads();
      if( max_threads > chunk_neighbors_scratch.thread.size() )
      {
        chunk_neighbors_scratch.thread.resize( max_threads );
      }
      
      GridChunkNeighborsHostWriteAccessor chunk_nbh( chunk_neighbors );
      
#     pragma omp parallel
      {        
        int tid = omp_get_thread_num();
        assert( tid>=0 && size_t(tid)<max_threads ); 
        auto& cell_a_particle_nbh = chunk_neighbors_scratch.thread[tid].cell_a_particle_nbh;
/*        
#       ifndef NDEBUG
        std::vector<unsigned int> cell_a_particle_chunk_count;
#       endif
*/
        GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) /*reduction(+:total_nbh_chunk)*/ )
        {
          size_t n_particles_a = cells[cell_a].size();

          //if( n_particles_a > cell_a_particle_nbh.size() )
          //{
            cell_a_particle_nbh.resize( n_particles_a );
          //}
          for(size_t i=0;i<n_particles_a;i++)
          {
            cell_a_particle_nbh[i].clear();
          }

          const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
          const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
          const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);

          ssize_t sgstart_a = sub_grid_start[cell_a];
          ssize_t sgsize_a = sub_grid_start[cell_a+1] - sgstart_a;
          ssize_t n_sub_cells_a = sgsize_a+1;
          ssize_t sgside_a = icbrt64( n_sub_cells_a );
          assert( sgside_a <= static_cast<ssize_t>(GRID_CHUNK_NBH_MAX_AMR_RES) );
          //double subcell_size_a = cell_size / sgside_a;

          ssize_t bstarti = std::max( loc_a.i-loc_max_gap , 0l ); 
          ssize_t bendi = std::min( loc_a.i+loc_max_gap , dims.i-1 );
          ssize_t bstartj = std::max( loc_a.j-loc_max_gap , 0l );
          ssize_t bendj = std::min( loc_a.j+loc_max_gap , dims.j-1 );
          ssize_t bstartk = std::max( loc_a.k-loc_max_gap , 0l );
          ssize_t bendk = std::min( loc_a.k+loc_max_gap , dims.k-1 );

          for(ssize_t loc_bk=bstartk;loc_bk<=bendk;loc_bk++)
          for(ssize_t loc_bj=bstartj;loc_bj<=bendj;loc_bj++)
          for(ssize_t loc_bi=bstarti;loc_bi<=bendi;loc_bi++)
          {
            IJK loc_b { loc_bi, loc_bj, loc_bk };
            ssize_t cell_b = grid_ijk_to_index( dims, loc_b );
            size_t n_particles_b = cells[cell_b].size();

            ssize_t sgstart_b = sub_grid_start[cell_b];
            ssize_t sgsize_b = sub_grid_start[cell_b+1] - sgstart_b;
            ssize_t n_sub_cells_b = sgsize_b+1;
            ssize_t sgside_b = icbrt64( n_sub_cells_b );
            assert( sgside_b <= static_cast<ssize_t>(GRID_CHUNK_NBH_MAX_AMR_RES) );
            // double subcell_size_b = cell_size / sgside_b;

            const auto* __restrict__ rx_b = cells[cell_b][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_b);
            const auto* __restrict__ ry_b = cells[cell_b][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_b);
            const auto* __restrict__ rz_b = cells[cell_b][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_b);

            IJK rloc_b = loc_b - loc_a; // relative (to cell b) position of cell a in grid
            uint16_t cell_b_enc = encode_cell_index( rloc_b );
            
            // what is the resolution pair combination ( local and neighbor cell respective resolutions)
            size_t res_pair_id = unique_pair_id( sgside_a-1 , sgside_b-1 );
            
            IJK rl = rloc_b;
            bool rev_ab = ( sgside_a > sgside_b );
            if( rev_ab ) { rl = IJK{ -rloc_b.i , -rloc_b.j , -rloc_b.k }; }
            bool rev_i = (rl.i<0);
            bool rev_j = (rl.j<0);
            bool rev_k = (rl.k<0);
            if( rev_i ) { rl.i = - rl.i; }
            if( rev_j ) { rl.j = - rl.j; }
            if( rev_k ) { rl.k = - rl.k; }
            
            // where in the grid pair cache is located the data for this cell pair
            const size_t cpoffset = res_pair_id * n_nbh_cell;
            const size_t block_index = rl.k*nbh_cell_side*nbh_cell_side + rl.j*nbh_cell_side + rl.i;
            const auto& cp = amr_grid_pairs.m_sub_cell_pairs[ cpoffset + block_index ];

            const size_t n_pairs = cp.m_pair_ab.size() / 2;
            const uint16_t* pair_ab = cp.m_pair_ab.data();
            const unsigned int A=rev_ab, B=!rev_ab;

            for(unsigned int scp=0;scp<n_pairs;scp++)
            {
              uint16_t ax = pair_ab[scp*2+A];
              unsigned int sca_i = ax & 31; ax = ax >> 5;
              assert( sca_i < sgside_a );
              unsigned int sca_j = ax & 31; ax = ax >> 5;
              assert( sca_j < sgside_a );
              unsigned int sca_k = ax;
              assert( sca_k < sgside_a );

              uint16_t bx = pair_ab[scp*2+B];
              unsigned int scb_i = bx & 31; bx = bx >> 5;
              assert( scb_i < sgside_b );
              unsigned int scb_j = bx & 31; bx = bx >> 5;
              assert( scb_j < sgside_b );
              unsigned int scb_k = bx;
              assert( scb_k < sgside_b );

              if( rev_i ) { sca_i = sgside_a-1-sca_i; scb_i = sgside_b-1-scb_i; }
              if( rev_j ) { sca_j = sgside_a-1-sca_j; scb_j = sgside_b-1-scb_j; }
              if( rev_k ) { sca_k = sgside_a-1-sca_k; scb_k = sgside_b-1-scb_k; }

              unsigned int sgindex_a = sg_cell_index( sgside_a , IJK{sca_i,sca_j,sca_k} , enable_z_order ); // grid_ijk_to_index( IJK{sgside_a,sgside_a,sgside_a} , IJK{sca_i,sca_j,sca_k} );
              unsigned int p_start_a = 0;
              unsigned int p_end_a = n_particles_a;
              if( sgindex_a > 0 ) { p_start_a = sub_grid_cells[sgstart_a+sgindex_a-1]; }
              if( sgindex_a < sgsize_a ) { p_end_a = sub_grid_cells[sgstart_a+sgindex_a]; }

              unsigned int sgindex_b = sg_cell_index( sgside_b , IJK{scb_i,scb_j,scb_k} , enable_z_order ); // grid_ijk_to_index( IJK{sgside_b,sgside_b,sgside_b} , IJK{scb_i,scb_j,scb_k} );
              unsigned int p_start_b = 0;
              unsigned int p_end_b = n_particles_b;
              if( sgindex_b > 0 ) { p_start_b = sub_grid_cells[sgstart_b+sgindex_b-1]; }
              if( sgindex_b < sgsize_b ) { p_end_b = sub_grid_cells[sgstart_b+sgindex_b]; }

              if( p_end_b > p_start_b )
              {
                for(unsigned int p_a=p_start_a;p_a<p_end_a;p_a++)
                {
                  for(unsigned int p_b=p_start_b;p_b<p_end_b;)
                  {
                    const Vec3d dr = { rx_a[p_a] - rx_b[p_b] , ry_a[p_a] - ry_b[p_b] , rz_a[p_a] - rz_b[p_b] };
                    double d2 = norm2( xform.transformCoord( dr ) );
                    if( nbh_filter(d2,max_dist2,cell_a,p_a,cell_b,p_b) )
                    {
                      unsigned int chunk_b = p_b >> cs_log2;
                      assert( chunk_b < std::numeric_limits<uint16_t>::max() );
                      std::pair<uint16_t,uint16_t> nbh_cell_chunk = { cell_b_enc , chunk_b };
                      cell_a_particle_nbh[p_a].push_back( nbh_cell_chunk );
                      p_b = ( chunk_b + 1 ) << cs_log2;
                    }
                    else
                    {
                      ++ p_b;
                    }
                  }
                }
              }
            }

          }


          //****************** fill in chunk lists here ******************************

          // temporary stream buffer
          auto& ccnbh = chunk_neighbors_scratch.thread[tid].encoded_stream;
          ccnbh.clear();

          /* optional stream indexing
           * Important note :
           * 1) neighbor stream (after particle offsets) cannot start with 1,0
           *    this would mean 1 neighbor cell whose encoded id would be 0, which is forbidden.
           * 2) particle offsets are given relative to nbh stream (not counting particle offsets themselves),
           *    are shifted by 1 and are encoded in 32-bits (two consecutive stream values for each offset), so the first
           *    offset will be 0 + 1 = 1, splitted in 2 16-bits words it gives 1,0
           * 3) a cell's neighbor stream data that would start with 1 and then 0 indicates it indeed contains particle offset
           * 4) the same apply if stream would start with 2,0 : encoded value of 0 for the first (of the 2 cells) is forbidden
           *    thus, we extend the previous so that the first integer is the number of 32-bits offset tables stored ahead of neighbors stream.
           * 4-bis) we limit number of tables to MAX_STREAM_OFFSET_TABLES
           */
          ssize_t offset_table_size = 0;
          unsigned int num_offset_tables = 0;
          if( build_particle_offset )
          {
            // first table has N+1 elements. it stores start offset of each particle's neighbor stream first is required to be 0, last will be total size of stream
            offset_table_size = (n_particles_a+1) * 2; 
            num_offset_tables = 1;
            // second table has N elements, it stores stream position where to stop/start to process half symmetric portion of neighbor list
            if( config.dual_particle_offset )
            {
              offset_table_size += n_particles_a * 2; 
              num_offset_tables = 2;
            }
          }
          if( n_particles_a > 0 )
          {
            ccnbh.assign( offset_table_size , 0 );
          }

          // build compact neighbor stream
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            // optional stream indexing
            if( build_particle_offset )
            {
              assert( ccnbh.size() >= offset_table_size );
              uint32_t offset = ccnbh.size() - offset_table_size + num_offset_tables;
              ccnbh[p_a*2+0] = offset ;
              ccnbh[p_a*2+1] = offset >> 16 ;
              [[maybe_unused]] const uint32_t* offset_table = reinterpret_cast<const uint32_t*>( ccnbh.data() );
              assert( offset_table[p_a] == offset );
              assert( p_a!=0 || ( ccnbh[0]==num_offset_tables && ccnbh[1]==0 ) );
            }
           
            // *** remove potential duplicates ***
            
            // removed because we know chunks are sorted, due to construction method (order of neighbor cells and sub grid pairs)
            std::sort( cell_a_particle_nbh[p_a].begin() , cell_a_particle_nbh[p_a].end() );
            
            unsigned int nbh_count = cell_a_particle_nbh[p_a].size();
            unsigned int nbh_count_nodup = 1;
            for(unsigned int nbh_i=1;nbh_i<nbh_count;nbh_i++)
            {
              if( cell_a_particle_nbh[p_a][nbh_i] != cell_a_particle_nbh[p_a][nbh_count_nodup-1] )
              {
                cell_a_particle_nbh[p_a][nbh_count_nodup] = cell_a_particle_nbh[p_a][nbh_i];
                ++ nbh_count_nodup;
              }
            }
            nbh_count_nodup = std::min( nbh_count_nodup , nbh_count );
            
            // adjust size to fit unique neighbor count                        
            cell_a_particle_nbh[p_a].resize( nbh_count_nodup );
            
            // initialize neighbor stream for particle p_a
            size_t cell_count_idx = ccnbh.size(); // place of neighbor cell counter in stream
            size_t chunk_count_idx = std::numeric_limits<size_t>::max();
            uint16_t last_cell = 0; // initialized with invalid value : 0 is not a correct encoded cell. minimum encoded cell is 1057 (2^10+2^5+2^0)
            if( ! config.random_access )
            {
              ccnbh.push_back(0); // neighbor cell counter initialized to 0
            }
            
            // encode stream for particle p_a
            uint32_t n_sym_nbh = 0;
            for( const auto& nbh : cell_a_particle_nbh[p_a] )
            {
              assert( nbh.first >= GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE );
              if( config.dual_particle_offset )
              {
                const int rel_i = int( nbh.first & 31 ) - 16;
                const int rel_j = int( (nbh.first>>5) & 31 ) - 16;
                const int rel_k = int( (nbh.first>>10) & 31 ) - 16;
                ssize_t cell_b = cell_a + ( ( ( rel_k * dims.j ) + rel_j ) * dims.i + rel_i );
                size_t p_b = int(nbh.second) << cs_log2;
                if( cell_b<cell_a || ( cell_b==cell_a && p_b<p_a ) )
                {
                  ++ n_sym_nbh;
                }
              }
              if( config.random_access )
              {
                ccnbh.push_back( nbh.first );
                ccnbh.push_back( nbh.second );
              }
              else
              {
                if( nbh.first!=last_cell )
                {
                  // start a new neighbor cell chunk
                  assert( ccnbh[cell_count_idx] < std::numeric_limits<uint16_t>::max() );
                  ++ ccnbh[cell_count_idx];
                  last_cell = nbh.first;
                  ccnbh.push_back( last_cell );
                  chunk_count_idx = ccnbh.size();
                  ccnbh.push_back(0);                
                }
                assert( ccnbh[chunk_count_idx] < std::numeric_limits<uint16_t>::max() );
                ++ ccnbh[chunk_count_idx];
                ccnbh.push_back( nbh.second );
              }
            }

            if( config.dual_particle_offset )
            {
              assert( ccnbh.size() >= offset_table_size );
              assert( /* n_sym_nbh>=0 && */ n_sym_nbh<=nbh_count_nodup );
              assert(( n_particles_a + 1 + p_a ) * 2 + 1 < offset_table_size );
              ccnbh[ ( n_particles_a + 1 + p_a ) * 2 + 0 ] = n_sym_nbh ;
              ccnbh[ ( n_particles_a + 1 + p_a ) * 2 + 1 ] = n_sym_nbh >> 16 ;
              [[maybe_unused]] const uint32_t* offset_table = reinterpret_cast<const uint32_t*>( ccnbh.data() + ( n_particles_a + 1 ) * 2 );
              assert( offset_table[p_a] == n_sym_nbh );
            }
          }

          // optional stream indexing
          // close main offset table with total size of stream.
          //  to calculate sub stream size for each particle
          if( n_particles_a>0 && build_particle_offset )
          {
            assert( n_particles_a*2+1 < offset_table_size );
            uint32_t offset = ccnbh.size() - offset_table_size + num_offset_tables;
            ccnbh[n_particles_a*2+0] = offset ;
            ccnbh[n_particles_a*2+1] = offset >> 16 ;
            [[maybe_unused]] const uint32_t* offset_table = reinterpret_cast<const uint32_t*>( ccnbh.data() );
            assert( offset_table[n_particles_a] == offset );
          }

          uint16_t * output_stream = chunk_nbh.allocate( cell_a , ccnbh.size() );
          std::memcpy( output_stream , ccnbh.data() , ccnbh.size() * sizeof(uint16_t) );
        }
        GRID_OMP_FOR_END
      }

      if( config.free_scratch_memory )
      {
        chunk_neighbors_scratch.thread.clear();
        chunk_neighbors_scratch.thread.shrink_to_fit();
      }

      chunk_neighbors.update_stream_pool_hint();
      ldbg << "Chunk neighbors next pre-alloc hint = "<<chunk_neighbors.m_stream_pool_hint <<", nb dyn alloc = "<<chunk_neighbors.m_nb_dyn_alloc<<std::endl;
    }

}

