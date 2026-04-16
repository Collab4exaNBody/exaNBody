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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <mpi.h>

namespace exanb
{

  /*
  Tracks multiple regions with contiguous, non-overlapping particle ID ranges.
  Algorithm guarantees:
  - Each tracked region gets a contiguous range of IDs
  - No particle can belong to multiple tracked regions (first match wins)
  - Particles outside all tracked regions get IDs after all tracked regions
  WARNING: Cannot conserve original id order or values, nor other tracked region id intervals
  */
  template<class GridT , class = AssertGridHasFields<GridT,field::_id> >
  class TrackRegionParticlesMultiple : public OperatorNode
  {  
    ADD_SLOT( MPI_Comm          , mpi    , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain            , domain , INPUT );
    ADD_SLOT( GridT             , grid   , INPUT_OUTPUT);
    
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT_OUTPUT );
    ADD_SLOT( std::vector<std::string>       , names , INPUT , REQUIRED );
    ADD_SLOT( std::vector<ParticleRegionCSG> , exprs , INPUT , OPTIONAL );

  public:

    inline std::string documentation() const override final
    {
      return R"EOF(
Reassigns particle IDs so that each tracked region receives its own contiguous block of
IDs (region 0 first, then region 1, etc.), and particles outside all tracked regions
receive IDs after the last region block. Each block is globally contiguous across MPI
ranks (via MPI_Exscan). All tracked regions are then registered in particle_regions under
their respective names with ID-range flags for fast membership tests.

When a particle matches more than one region expression, it is assigned to the first
matching region (first-match-wins). The algorithm does NOT preserve original ID values or
ordering of any particle.

Example:
# First define regions
  - particle_regions:
      - BOTTOMBOX:
          bounds: [ [ -200 ang , -200 ang , -200 ang ] , [ 200 ang , 200 ang , 15 ang ] ]
      - TOPBOX:
          bounds: [ [ -200 ang , -200 ang ,  135 ang ] , [ 200 ang , 200 ang , 200 ang ] ]
# Then define these regions as "tracked" regions.
  - track_region_particles_multiple:
      exprs: [ "BOTTOMBOX" , "TOPBOX" ]
      names: [ "BAS"       , "HAUT"  ]

If you only need to track a single region, track_region_particles is simpler.
)EOF";
    }
    
    inline void execute() override final
    {    
      // Validate inputs
      if( !particle_regions.has_value() )
      {
        fatal_error() << "particle_regions has no value" << std::endl;
      }
      
      size_t n_tracked_regions = names->size();
      if( n_tracked_regions == 0 )
      {
        fatal_error() << "At least one region name must be specified" << std::endl;
      }
      
      // Build region name map
      std::map< std::string , unsigned int > region_name_map;
      unsigned int n_existing_regions = particle_regions->size();
      for(unsigned int i=0;i<n_existing_regions;i++)
      {
        const auto & r = particle_regions->at(i);
        region_name_map[ r.name() ] = i;
      }

      // Prepare CSG expressions for each tracked region
      std::vector<ParticleRegionCSGShallowCopy> prcsg_list(n_tracked_regions);
      
      if( exprs.has_value() && exprs->size() != n_tracked_regions )
      {
        fatal_error() << "Number of expressions (" << exprs->size() 
                     << ") must match number of region names (" << n_tracked_regions << ")" << std::endl;
      }
      
      for(size_t i = 0; i < n_tracked_regions; i++)
      {
        if( exprs.has_value() && i < exprs->size() )
        {
          if( (*exprs)[i].m_nb_operands == 0 )
          {
            (*exprs)[i].build_from_expression_string( particle_regions->data() , particle_regions->size() );
          }
          prcsg_list[i] = (*exprs)[i];
        }
      }

      // MPI setup
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&nprocs);

      const auto xform = domain->xform();
      size_t n_cells = grid->number_of_cells();
      auto cells = grid->cells();

      // Count particles in each tracked region and outside all regions
      std::vector<unsigned long long> n_particles_per_region(n_tracked_regions, 0);
      unsigned long long n_particles_outside_all_regions = 0;
      
#     pragma omp parallel
      {
        std::vector<unsigned long long> local_counts(n_tracked_regions, 0);
        unsigned long long local_outside = 0;
        
#       pragma omp for schedule(dynamic)
        for(size_t cell_i=0;cell_i<n_cells;cell_i++)
        {
          if( grid->is_ghost_cell(cell_i) ) continue;
          
          size_t n_particles = cells[cell_i].size();
          for(size_t p=0;p<n_particles;p++)
          {
            const unsigned long long id = cells[cell_i][field::id][p];
            Vec3d pos = xform * Vec3d{cells[cell_i][field::rx][p],cells[cell_i][field::ry][p],cells[cell_i][field::rz][p]};
            
            // Check which region this particle belongs to (first match wins)
            bool found = false;
            for(size_t r=0; r<n_tracked_regions && !found; r++)
            {
              if( prcsg_list[r].contains( pos , id ) )
              {
                local_counts[r]++;
                found = true;
              }
            }
            if(!found)
            {
              local_outside++;
            }
          }
        }
        
#       pragma omp critical
        {
          for(size_t r=0; r<n_tracked_regions; r++)
          {
            n_particles_per_region[r] += local_counts[r];
          }
          n_particles_outside_all_regions += local_outside;
        }
      }

      // Global reduction for all region counts
      std::vector<unsigned long long> total_particles_per_region(n_tracked_regions, 0);
      MPI_Allreduce(n_particles_per_region.data(), total_particles_per_region.data(), 
                    n_tracked_regions, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);

      // Calculate ID offsets for each region using MPI_Exscan
      std::vector<unsigned long long> region_id_offsets(n_tracked_regions, 0);
      MPI_Exscan(n_particles_per_region.data(), region_id_offsets.data(), 
                 n_tracked_regions, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);
      
      // Adjust offsets to create contiguous ranges across all regions
      unsigned long long cumulative_offset = 0;
      for(size_t r=0; r<n_tracked_regions; r++)
      {
        region_id_offsets[r] += cumulative_offset;
        cumulative_offset += total_particles_per_region[r];
      }

      // Calculate offset for particles outside all tracked regions
      unsigned long long other_id_offset = 0;
      MPI_Exscan(&n_particles_outside_all_regions, &other_id_offset, 1, 
                 MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);
      other_id_offset += cumulative_offset; // Start after all tracked regions

      // Atomic counters for thread-safe ID assignment
      std::vector<std::atomic<uint64_t>> next_region_id(n_tracked_regions);
      for(size_t r=0; r<n_tracked_regions; r++)
      {
        next_region_id[r].store(region_id_offsets[r], std::memory_order_relaxed);
      }
      std::atomic<uint64_t> next_other_id = other_id_offset;

      // Reassign particle IDs
#     pragma omp parallel for schedule(dynamic)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        if( grid->is_ghost_cell(cell_i) ) continue;
        
        size_t n_particles = cells[cell_i].size();
        for(size_t p=0;p<n_particles;p++)
        {
          const unsigned long long id = cells[cell_i][field::id][p];
          Vec3d pos = xform * Vec3d{cells[cell_i][field::rx][p],cells[cell_i][field::ry][p],cells[cell_i][field::rz][p]};
          
          // Assign ID based on region membership (first match wins)
          bool found = false;
          for(size_t r=0; r<n_tracked_regions && !found; r++)
          {
            if( prcsg_list[r].contains( pos , id ) )
            {
              cells[cell_i][field::id][p] = next_region_id[r].fetch_add(1, std::memory_order_relaxed);
              found = true;
            }
          }
          if(!found)
          {
            cells[cell_i][field::id][p] = next_other_id.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }

      // Add tracked regions to particle_regions structure
      unsigned long long current_start = 0;
      for(size_t r=0; r<n_tracked_regions; r++)
      {
        ParticleRegion region = {};
        region.set_name( (*names)[r] );
        region.m_id_start = current_start;
        region.m_id_end = current_start + total_particles_per_region[r];
        region.m_id_range_flag = true;

        ldbg << "add tracking region " << region.name() << " with " 
             << (region.m_id_end - region.m_id_start) << " particles, "
             << "id range: [" << region.m_id_start << ", " << region.m_id_end << ")" << std::endl;

        particle_regions->push_back( region );
        current_start = region.m_id_end;
      }
    }

  };

  template<class GridT> using TrackRegionParticlesMultipleTmpl = TrackRegionParticlesMultiple< GridT >;

  // === register factories ===  
  ONIKA_AUTORUN_INIT(track_region_particles_multiple)
  {
    OperatorNodeFactory::instance()->register_factory( "track_region_particles_multiple", make_grid_variant_operator<TrackRegionParticlesMultipleTmpl> );
  }

}

