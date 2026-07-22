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

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_iterator.h>

#include <mpi.h>
#include <atomic>
#include <vector>
#include <cstdint>

namespace exanb
{

  template<class GridT, class = AssertGridHasFields<GridT, field::_id> >
  class RemoveCloseParticles : public OperatorNode
  {
    ADD_SLOT( MPI_Comm           , mpi             , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT              , grid            , INPUT_OUTPUT );
    ADD_SLOT( GridChunkNeighbors , chunk_neighbors , INPUT , GridChunkNeighbors{} );
    ADD_SLOT( double             , distance        , INPUT , REQUIRED , DocString{"Distance threshold. Whenever two particles are closer than this, the one with the higher particle id is removed."} );

  public:

    inline std::string documentation() const override final
    {
      return R"EOF(
Deletes particles that are closer to another particle than a given threshold distance.

For every too-close pair, the particle with the higher id is the one removed.
This tie-break is symmetric and needs no cross-rank coordination: each rank
only ever deletes particles it owns, based on its own neighbor list (which
includes ghost-mirrored particles owned by other ranks). Only non-ghost cells
are affected (ghost particles are rebuilt by the next ghost update anyway).
Particle ids are NOT renumbered; chain with 'compact_particle_ids' afterwards
if a dense [0,N) numbering is required.

Requires 'chunk_neighbors' to have been built earlier in the pipeline with a
search distance greater than or equal to 'distance', and NOT in
half_symmetric mode (the exaNBody default), since every particle needs to
see its full neighbor list to decide on its own whether it must be removed.

A single pass may not fully resolve chains of mutually close particles
(A close to B close to C): re-run the operator (or loop it) until 'removed'
converges to 0 if that matters for your use case.

Usage example:

# delete particles that are closer than 0.5 to another particle
  - remove_close_particles:
      distance: 0.5
)EOF";
    }

    inline void execute() override final
    {
      // no early return on zero local particles here: every rank must reach the
      // MPI_Allreduce below regardless, or ranks that locally have no particles
      // would desync the collective call from ranks that do.
      if( chunk_neighbors->number_of_cells() != grid->number_of_cells() )
      {
        fatal_error() << "remove_close_particles: chunk_neighbors is not built (or stale) for this grid."
                       << " Place this operator after neighbor lists have been computed"
                       << " (e.g. after 'init_particles' / inside '+first_iteration')." << std::endl;
      }

      auto cells = grid->cells();
      using CellT = std::remove_cv_t< std::remove_reference_t< decltype(cells[0]) > >;
      const IJK dims = grid->dimension();
      const ssize_t gl = grid->ghost_layers();
      const IJK gstart { gl, gl, gl };
      const IJK gend = dims - IJK{ gl, gl, gl };
      const IJK gdims = gend - gstart;
      const auto & cell_allocator = grid->cell_allocator();

      const double threshold2 = (*distance) * (*distance);
      std::vector< std::vector<uint8_t> > remove_flag( grid->number_of_cells() );

      ChunkParticleNeighborsIterator<CellT> chunk_nbh_it_in = { grid->cells(), chunk_neighbors->data(), dims, chunk_neighbors->m_chunk_size };

      // --- pass 1 : read-only decision pass ---
      // A particle marks itself if it has a closer-than-threshold neighbor with a
      // smaller id. Neighbor lists are full (not half_symmetric), so every particle
      // sees this on its own: no need to know the other side's decision. This makes
      // it safe to run in parallel over cells while other threads concurrently read
      // (but never write) neighboring cells' particle data.
#     pragma omp parallel
      {
        auto chunk_nbh_it = chunk_nbh_it_in;
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
          const size_t np = cells[cell_i].size();
          remove_flag[cell_i].assign( np, 0 );
          if( np > 0 )
          {
            const double* __restrict__ rx_a = cells[cell_i][field::rx];
            const double* __restrict__ ry_a = cells[cell_i][field::ry];
            const double* __restrict__ rz_a = cells[cell_i][field::rz];
            const uint64_t* __restrict__ id_a = cells[cell_i][field::id];

            chunk_nbh_it.start_cell( cell_i, np );
            for(size_t p_a=0;p_a<np;p_a++)
            {
              chunk_nbh_it.start_particle( p_a );
              bool remove = false;
              while( ! chunk_nbh_it.end() )
              {
                size_t cell_b=0, p_b=0;
                chunk_nbh_it.get_nbh( cell_b, p_b );
                const uint64_t id_b = cells[cell_b][field::id][p_b];
                if( id_b < id_a[p_a] )
                {
                  const double dx = cells[cell_b][field::rx][p_b] - rx_a[p_a];
                  const double dy = cells[cell_b][field::ry][p_b] - ry_a[p_a];
                  const double dz = cells[cell_b][field::rz][p_b] - rz_a[p_a];
                  if( dx*dx + dy*dy + dz*dz < threshold2 ) { remove = true; }
                }
                chunk_nbh_it.next();
              }
              remove_flag[cell_i][p_a] = remove ? 1 : 0;
            }
          }
        }
        GRID_OMP_FOR_END
      }

      // --- pass 2 : in-place compaction (swap-with-tail, same scheme as remove_particles) ---
      std::atomic<uint64_t> n_removed = 0;
#     pragma omp parallel
      {
        std::vector<int32_t> particle_pack;
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
          int32_t n = static_cast<int32_t>( cells[cell_i].size() );
          if( n > 0 )
          {
            particle_pack.resize( n );
            for(int32_t j=0;j<n;j++) { particle_pack[j] = j; }

            int32_t n_out = 0;
            for(int32_t j=0;j<n;j++)
            {
              if( remove_flag[cell_i][j] ) { particle_pack[j] = -1; ++n_out; }
            }

            if( n_out > 0 )
            {
              for(int32_t j=n-1;j>=0;j--)
              {
                if( particle_pack[j] == -1 )
                {
                  if( j < n-1 ) { particle_pack[j] = particle_pack[n-1]; }
                  --n;
                }
              }
              for(int32_t j=0;j<n;j++)
              {
                if( j != particle_pack[j] ) { cells[cell_i].set_tuple( j , cells[cell_i][ particle_pack[j] ] ); }
              }
              cells[cell_i].resize( n, cell_allocator );
              n_removed.fetch_add( static_cast<uint64_t>(n_out), std::memory_order_relaxed );
            }
          }
        }
        GRID_OMP_FOR_END
      }

      grid->rebuild_particle_offsets();

      unsigned long long local_removed = n_removed.load();
      unsigned long long global_removed = 0;
      MPI_Allreduce( &local_removed, &global_removed, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi );

      ldbg << "remove_close_particles: distance="<<(*distance)<<" removed="<<global_removed<<std::endl;
    }

  };

  template<class GridT> using RemoveCloseParticlesTmpl = RemoveCloseParticles<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(remove_close_particles)
  {
    OperatorNodeFactory::instance()->register_factory( "remove_close_particles", make_grid_variant_operator< RemoveCloseParticlesTmpl > );
  }

}
