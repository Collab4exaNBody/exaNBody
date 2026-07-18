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
#include <exanb/core/particle_type_id.h>

#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_localized_filter.h>
#include <exanb/grid_cell_particles/particle_random_selection.h>

#include <mpi.h>
#include <atomic>
#include <string>
#include <cstdint>

namespace exanb
{

  template<class GridT, class = AssertGridHasFields<GridT, field::_id, field::_type> >
  class SetParticleType : public OperatorNode
  {
    ADD_SLOT( MPI_Comm          , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT             , grid              , INPUT_OUTPUT );
    ADD_SLOT( Domain            , domain            , INPUT );

    ADD_SLOT( ParticleTypeMap   , particle_type_map , INPUT , OPTIONAL , DocString{"Maps type names to internal integer ids. Required in order to resolve 'type' by name."} );
    ADD_SLOT( std::string       , type              , INPUT , REQUIRED , DocString{"Name of the new particle type assigned to selected particles (looked up in particle_type_map)."} );

    ADD_SLOT( ParticleRegions   , particle_regions  , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region            , INPUT_OUTPUT , OPTIONAL , DocString{"Restricts candidate particles to this region (boolean expression over named particle_regions). If absent, the whole local domain is considered."} );

    ADD_SLOT( double            , fraction          , INPUT , OPTIONAL , DocString{"Fraction (0..1) of eligible particles to convert, drawn at random. Mutually exclusive with 'count'."} );
    ADD_SLOT( long               , count             , INPUT , OPTIONAL , DocString{"Exact number of eligible particles to convert, drawn at random, globally across all MPI ranks. Mutually exclusive with 'fraction'."} );
    ADD_SLOT( long               , seed              , INPUT , 0 , DocString{"Seed for the deterministic random selection. Same seed + same particle ids => same selection, regardless of rank/thread count."} );
    ADD_SLOT( bool               , ghost             , INPUT , false , DocString{"If true, ghost cells are processed too (rarely useful: ghosts get overwritten on the next ghost update)."} );

  public:

    inline std::string documentation() const override final
    {
      return R"EOF(
Randomly reassigns the type of a subset of particles.

Candidate ("eligible") particles can optionally be restricted to a spatial
region via 'region' (a boolean expression over named 'particle_regions').
Among the eligible particles, the ones actually converted are chosen either:
  - all of them                     (neither 'fraction' nor 'count' given)
  - a random fraction of them        ('fraction': 0..1, expected count only)
  - an exact random number of them   ('count': exact, global across ranks)

Selection is deterministic given 'seed' and particle ids: re-running the
operator on the same grid with the same seed reproduces the same selection,
independently of the number of MPI ranks or OpenMP threads.

Usage examples:

# convert every particle inside a region to type B
  - set_particle_type:
      region: INCLUSION
      type: B

# convert ~20% of the particles inside a region to type B
  - set_particle_type:
      region: INCLUSION
      type: B
      fraction: 0.2
      seed: 42

# convert exactly 5000 particles anywhere in the domain to type B
  - set_particle_type:
      type: B
      count: 5000
      seed: 42
)EOF";
    }

    inline void execute() override final
    {
      if( fraction.has_value() && count.has_value() )
      {
        fatal_error() << "set_particle_type: 'fraction' and 'count' are mutually exclusive" << std::endl;
      }
      if( ! particle_type_map.has_value() )
      {
        fatal_error() << "set_particle_type: particle_type_map is undefined, cannot resolve type name '"<<(*type)<<"'" << std::endl;
      }
      const auto it = particle_type_map->find( *type );
      if( it == particle_type_map->end() )
      {
        fatal_error() << "set_particle_type: unknown particle type '"<<(*type)<<"'" << std::endl;
      }
      const ParticleTypeInt new_type_id = static_cast<ParticleTypeInt>( it->second );

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
      ssize_t gl = 0;
      if( ! *ghost ) { gl = grid->ghost_layers(); }
      const IJK gstart { gl, gl, gl };
      const IJK gend = dims - IJK{ gl, gl, gl };
      const IJK gdims = gend - gstart;

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

      std::atomic<uint64_t> n_converted = 0;
#     pragma omp parallel
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
              if( particle_random_key(id,seedv) < th.tau )
              {
                cells[cell_i][field::type][p] = new_type_id;
                ++local_n;
              }
            }
          }
          n_converted.fetch_add( local_n, std::memory_order_relaxed );
        }
        GRID_OMP_FOR_END
      }

      unsigned long long local_converted = n_converted.load();
      unsigned long long global_converted = 0;
      MPI_Allreduce( &local_converted, &global_converted, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi );

      ldbg << "set_particle_type: eligible="<<th.global_eligible<<" target="<<th.global_target
           <<" converted="<<global_converted<<" -> type '"<<(*type)<<"' (id="<<int(new_type_id)<<")"<<std::endl;
    }

  };

  template<class GridT> using SetParticleTypeTmpl = SetParticleType<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(set_particle_type)
  {
    OperatorNodeFactory::instance()->register_factory( "set_particle_type", make_grid_variant_operator< SetParticleTypeTmpl > );
  }

}
