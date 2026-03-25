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

#include <onika/scg/operator.h>
#include <onika/thread.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/grid_cell_particles/particle_region.h>

#include <vector>
#include <algorithm>

namespace exanb
{
  struct FilterParticleRegionESNullOp {
    // Future Devs
  };

  template<class GridT, typename FilterParticleRegionES>
  inline void filter_particle_region_cells(const Domain& domain, GridT& grid,
                                           ParticleRegionCSGShallowCopy& prcsg,
                                           FilterParticleRegionES& optional_storage)
  {
    const auto & cell_allocator = grid.cell_allocator();
    auto cells = grid.cells();
    IJK dims = grid.dimension();

#pragma omp parallel for schedule(dynamic)
    for(size_t cell_i=0 ; cell_i<cells->size() ; cell_i++)
    {
      auto& cell = cells[cell_i];
      const double* __restrict__ rx = cell[field::rx];
      const double* __restrict__ ry = cell[field::ry];
      const double* __restrict__ rz = cell[field::rz];
      const uint64_t* __restrict__ id = cell[field::id];
      int n = cell.size();
      if (n==0) {
        continue;
      }
      int rm_size = 0;
      // test if a particle is filtered in the cell
      for(int p_i=0;p_i<n;p_i++)
      {
        Vec3d r{rx[p_i],ry[p_i],rz[p_i]};
        if(prcsg.contains(r, id[p_i]))
        {
          rm_size++;
        }
      }
/*
      if (rm_size == n)
      {
        lout << "Clear " << rm_size << " particles." << std::endl;
        cell.clear();
      }
      else*/ if (rm_size > 0)
      {
        lout << "Remove " << rm_size << " particles / " << n << std::endl;
        int new_size = n;
        for (int p_i=n-1;p_i>=0;p_i--)
        {
          Vec3d r{rx[p_i],ry[p_i],rz[p_i]};
          if(prcsg.contains(r, id[p_i]))  // We replace the particles to be removed with valid particles.
          {
            new_size--;
            cell[p_i] = cell[new_size];  // new_size idx = "old" last element
          }
        }
        if (new_size != n - rm_size) {
          std::cout << "Error, filter step failled" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        assert(new_size == n - rm_size);
        lout << "Resize " << new_size << std::endl;
        cell.resize(new_size, cell_allocator);

      }
      // recheck
      n = cell.size();
      int error = 0; 
      for(int p_i=0;p_i<n;p_i++)
      {
        Vec3d r{rx[p_i],ry[p_i],rz[p_i]};
        if(prcsg.contains(r, id[p_i]))
        {
          error++;
        }
      }
      lout << "Number of error <on rank 0, cell " << cell_i << ">: " << error << std::endl; 
    }
    grid.rebuild_particle_offsets();
  }

  template<class GridT>
  class FilterParticleRegion : public OperatorNode
  { 
    ADD_SLOT( Domain            , domain           , INPUT );
    ADD_SLOT( GridT             , grid             , INPUT_OUTPUT );
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT, REQUIRED);
    ADD_SLOT( ParticleRegionCSG , region           , INPUT, REQUIRED);

   public:
    inline std::string documentation() const final
    {
      return R"EOF(
              This operator removes particle(s) included in a region.

              Parameter:

                - region [INPUT/REQUIRED]: Specifies the geographic areas or ID ranges where the particles will be removed.

              YAML example:

                filter_particle_region:
                   - region: REGION1 and REGION2
             )EOF";
    }

    inline void execute () final
    {
      if (!particle_regions.has_value())
      {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }

      if (region->m_nb_operands == 0)
      {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      ParticleRegionCSGShallowCopy prcsg = *region;
      FilterParticleRegionESNullOp optional;
      filter_particle_region_cells(*domain, *grid, prcsg, optional);
    }    
  };
}
