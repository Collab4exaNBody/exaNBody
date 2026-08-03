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
#include <exanb/core/domain.h>
#include <exanb/core/xform.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_localized_filter.h>
#include <exanb/grid_cell_particles/particle_random_selection.h>

#include <mpi.h>
#include <atomic>
#include <vector>
#include <cstdint>

namespace exanb
{

  template<class GridT, class = AssertGridHasFields<GridT, field::_id> >
  class RemoveParticles : public OperatorNode
  {
    ADD_SLOT( MPI_Comm          , mpi              , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT             , grid             , INPUT_OUTPUT );
    ADD_SLOT( Domain            , domain           , INPUT );

    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL , DocString{"Restricts candidate particles to this region (boolean expression over named particle_regions). If absent, the whole local domain is considered."} );

    ADD_SLOT( double            , fraction         , INPUT , OPTIONAL , DocString{"Fraction (0..1) of eligible particles to delete, drawn at random. Mutually exclusive with 'count'."} );
    ADD_SLOT( long               , count            , INPUT , OPTIONAL , DocString{"Exact number of eligible particles to delete, drawn at random, globally across all MPI ranks. Mutually exclusive with 'fraction'."} );
    ADD_SLOT( long               , seed             , INPUT , 0 , DocString{"Seed for the deterministic random selection. Same seed + same particle ids => same selection, regardless of rank/thread count."} );

  public:

    inline std::string documentation() const override final
    {
      return R"EOF(
Deletes a subset of particles from the grid.

Candidate ("eligible") particles can optionally be restricted to a spatial
region via 'region' (a boolean expression over named 'particle_regions').
Among the eligible particles, the ones actually deleted are chosen either:
  - all of them                     (neither 'fraction' nor 'count' given)
  - a random fraction of them        ('fraction': 0..1, expected count only)
  - an exact random number of them   ('count': exact, global across ranks)

Selection is deterministic given 'seed' and particle ids. Only non-ghost
cells are affected (ghost particles are rebuilt by the next ghost update
anyway). Particle ids are NOT renumbered by this operator: deleting
particles leaves holes in the id numbering. Chain with 'compact_particle_ids'
afterwards if a dense [0,N) numbering is required.

Usage examples:

# delete every particle inside a region
  - remove_particles:
      region: CAVITY

# delete a random 5% of the particles anywhere in the domain
  - remove_particles:
      fraction: 0.05
      seed: 7

# delete exactly 1000 random particles inside a region
  - remove_particles:
      region: CAVITY
      count: 1000
      seed: 7
)EOF";
    }

    inline void execute() override final
    {
      if( fraction.has_value() && count.has_value() )
      {
        fatal_error() << "remove_particles: 'fraction' and 'count' are mutually exclusive" << std::endl;
      }

      PartcileLocalizedFilter<GridT,LinearXForm> particle_filter = { *grid, { domain->xform() } };
      particle_filter.initialize_from_optional_parameters( particle_regions.get_pointer(), region.get_pointer() );

      ParticleSelectionCount sel;
      sel.seed = static_cast<uint64_t>( *seed );
      if( fraction.has_value() )      { sel.mode = ParticleSelectionCount::FRACTION; sel.fraction = *fraction; }
      else if( count.has_value() )    { sel.mode = ParticleSelectionCount::COUNT;    sel.count    = static_cast<uint64_t>( *count ); }
      else                            { sel.mode = ParticleSelectionCount::ALL; }
      const uint64_t seedv = sel.seed;

      auto cells = grid->cells();
      const IJK dims = grid->dimension();
      const ssize_t gl = grid->ghost_layers();
      const IJK gstart { gl, gl, gl };
      const IJK gend = dims - IJK{ gl, gl, gl };
      const IJK gdims = gend - gstart;
      const auto & cell_allocator = grid->cell_allocator();

      // only non-ghost cells are ever considered: ghost cells mirror another
      // rank's owned particles and are fully rebuilt by the next ghost update.
      auto is_eligible = [&]( size_t cell_i, size_t p ) -> bool
      {
        const Vec3d r = { cells[cell_i][field::rx][p], cells[cell_i][field::ry][p], cells[cell_i][field::rz][p] };
        const uint64_t id = cells[cell_i][field::id][p];
        return particle_filter( r, id );
      };

      auto count_eligible = [&]() -> uint64_t
      {
        std::atomic<uint64_t> n = 0;
#       pragma omp parallel
        {
          GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
          {
            const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
            const size_t np = cells[cell_i].size();
            uint64_t local_n = 0;
            for(size_t p=0;p<np;p++) { if( is_eligible(cell_i,p) ) ++local_n; }
            n.fetch_add( local_n, std::memory_order_relaxed );
          }
          GRID_OMP_FOR_END
        }
        return n.load();
      };

      auto count_below = [&]( double tau ) -> uint64_t
      {
        std::atomic<uint64_t> n = 0;
#       pragma omp parallel
        {
          GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
          {
            const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
            const size_t np = cells[cell_i].size();
            uint64_t local_n = 0;
            for(size_t p=0;p<np;p++)
            {
              if( is_eligible(cell_i,p) )
              {
                const uint64_t id = cells[cell_i][field::id][p];
                if( particle_random_key(id,seedv) < tau ) ++local_n;
              }
            }
            n.fetch_add( local_n, std::memory_order_relaxed );
          }
          GRID_OMP_FOR_END
        }
        return n.load();
      };

      const auto th = compute_particle_selection_threshold( *mpi, sel, count_eligible, count_below );

      // --- selection + in-place compaction pass ---
      // For each cell: mark selected (to-delete) particle slots with -1 in a
      // "particle_pack" permutation array, then, scanning from the tail down,
      // fill each hole with the last still-valid slot (this is the same
      // swap-with-tail compaction used when particles leave a cell in
      // move_particles_across_cells.h). Finally shrink the cell to its new size.
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
              if( is_eligible(cell_i,j) )
              {
                const uint64_t id = cells[cell_i][field::id][j];
                if( particle_random_key(id,seedv) < th.tau )
                {
                  particle_pack[j] = -1;
                  ++n_out;
                }
              }
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

      ldbg << "remove_particles: eligible="<<th.global_eligible<<" target="<<th.global_target
           <<" removed="<<global_removed<<std::endl;
    }

  };

  template<class GridT> using RemoveParticlesTmpl = RemoveParticles<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(remove_particles)
  {
    OperatorNodeFactory::instance()->register_factory( "remove_particles", make_grid_variant_operator< RemoveParticlesTmpl > );
  }

}
