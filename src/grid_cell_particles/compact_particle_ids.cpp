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

#include <mpi.h>
#include <vector>
#include <cstdint>

namespace exanb
{

  template<class GridT, class = AssertGridHasFields<GridT, field::_id> >
  class CompactParticleIds : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi  , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT    , grid , INPUT_OUTPUT );

  public:

    inline std::string documentation() const override final
    {
      return R"EOF(
Renumbers every (non-ghost, locally owned) particle's id so that ids form a
dense, contiguous range [0, N) across the whole simulation (N = total number
of particles, summed over all MPI ranks), with no gaps.

This is typically needed after operators that remove particles (such as
'remove_particles'), which leave holes in the id numbering.

Numbering is fully deterministic: particles are numbered in increasing cell
index order within a rank, and ranks are ordered by MPI rank (via
MPI_Exscan), so re-running this operator on an unchanged grid always
produces the same ids, regardless of the number of OpenMP threads used.

WARNING: this reassigns particle identity. Do not use it if anything else
in the simulation still refers to the previous ids (e.g. bonded/molecular
topology tables, external post-processing keyed by id, or dump files meant
to be diffed against a previous state by id). Typically used once right
after 'remove_particles', before any such id-dependent data is built.

Example:
  - remove_particles:
      region: CAVITY
      fraction: 0.1
  - compact_particle_ids
)EOF";
    }

    inline void execute() override final
    {
      auto cells = grid->cells();
      const IJK dims = grid->dimension();
      const ssize_t gl = grid->ghost_layers();
      const IJK gstart { gl, gl, gl };
      const IJK gend = dims - IJK{ gl, gl, gl };
      const IJK gdims = gend - gstart;
      const size_t n_cells = grid->number_of_cells();

      // per-cell particle counts, indexed by linear cell index so that
      // numbering order only depends on the (fixed, deterministic) grid
      // layout, never on thread scheduling.
      std::vector<uint64_t> cell_offset( n_cells, 0 );

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
          cell_offset[cell_i] = cells[cell_i].size();
        }
        GRID_OMP_FOR_END
      }

      // sequential exclusive prefix sum over cells (one entry per cell,
      // negligible cost compared to the per-particle work below)
      uint64_t local_n_particles = 0;
      for(size_t i=0;i<n_cells;i++)
      {
        const uint64_t c = cell_offset[i];
        cell_offset[i] = local_n_particles;
        local_n_particles += c;
      }

      uint64_t rank_id_start = 0;
      MPI_Exscan( &local_n_particles, &rank_id_start, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi );
      // (MPI_Exscan leaves the result undefined on rank 0; rank_id_start
      // was pre-initialized to 0, which is exactly what we want there)

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          const size_t cell_i = grid_ijk_to_index( dims, loc + gstart );
          const size_t np = cells[cell_i].size();
          uint64_t next_id = rank_id_start + cell_offset[cell_i];
          for(size_t p=0;p<np;p++)
          {
            cells[cell_i][field::id][p] = next_id++;
          }
        }
        GRID_OMP_FOR_END
      }

      uint64_t global_n_particles = 0;
      MPI_Allreduce( &local_n_particles, &global_n_particles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi );

      ldbg << "compact_particle_ids: renumbered "<<global_n_particles<<" particles into [0,"<<global_n_particles<<")"<<std::endl;
    }

  };

  template<class GridT> using CompactParticleIdsTmpl = CompactParticleIds<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(compact_particle_ids)
  {
    OperatorNodeFactory::instance()->register_factory( "compact_particle_ids", make_grid_variant_operator< CompactParticleIdsTmpl > );
  }

}
