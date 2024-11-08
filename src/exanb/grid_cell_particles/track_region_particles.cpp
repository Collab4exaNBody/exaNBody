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
  WARNING: only one region can be tracked by now. algothim works and guarantees contiguous particle ids all over the domain,
  BUT it cannot conserve original id order or values, nor other tracked region id intervals
  */
  template<class GridT , class = AssertGridHasFields<GridT,field::_id> >
  class TrackRegionParticles : public OperatorNode
  {  
    ADD_SLOT( MPI_Comm          , mpi    , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain            , domain , INPUT );
    ADD_SLOT( GridT             , grid   , INPUT_OUTPUT);
    
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT_OUTPUT );
    ADD_SLOT( std::string       , name , INPUT , REQUIRED );
    ADD_SLOT( ParticleRegionCSG , expr , INPUT , OPTIONAL );

  public:
    inline void execute() override final
    {    
      std::map< std::string , unsigned int > region_name_map;
      unsigned int n_regions = particle_regions->size();      
      for(unsigned int i=0;i<n_regions;i++)
      {
        const auto & r = particle_regions->at(i);
        region_name_map[ r.name() ] = i;
      }

      ParticleRegionCSGShallowCopy prcsg;
      if( !particle_regions.has_value() )
      {
        fatal_error() << "region is defined, but particle_regions has no value" << std::endl;
      }        
      if( expr->m_nb_operands==0 )
      {
        expr->build_from_expression_string( particle_regions->data() , particle_regions->size() );
      }
      prcsg = *expr;

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&nprocs);

      const auto xform = domain->xform();
      size_t n_cells = grid->number_of_cells();
      auto cells = grid->cells();

      unsigned long long max_id = 0;
      unsigned long long n_particles_in_region = 0;
      unsigned long long n_particles_outside_region = 0;
#     pragma omp parallel for schedule(dynamic) reduction(+:n_particles_in_region,n_particles_outside_region) reduction(max:max_id)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++) if( ! grid->is_ghost_cell(cell_i) )
      {
        size_t n_particles = cells[cell_i].size();
        for(size_t p=0;p<n_particles;p++)
        {
          const unsigned long long id = cells[cell_i][field::id][p];
          max_id = std::max( max_id , id );
          Vec3d pos = xform * Vec3d{cells[cell_i][field::rx][p],cells[cell_i][field::ry][p],cells[cell_i][field::rz][p]};
          if( prcsg.contains( pos , id ) ) { ++ n_particles_in_region; }
          else { ++ n_particles_outside_region; }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE,&max_id,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);

      unsigned long long total_particles_in_region = 0;
      MPI_Allreduce(&n_particles_in_region,&total_particles_in_region,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      
      unsigned long long region_id_offset = 0;
      MPI_Exscan( &n_particles_in_region , &region_id_offset , 1 , MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);

      unsigned long long other_id_offset = 0;
      MPI_Exscan( &n_particles_outside_region , &other_id_offset , 1 , MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      other_id_offset += total_particles_in_region;

      std::atomic<uint64_t> next_region_id = region_id_offset;
      std::atomic<uint64_t> next_other_id = other_id_offset;

#     pragma omp parallel for schedule(dynamic)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++) if( ! grid->is_ghost_cell(cell_i) )
      {
        size_t n_particles = cells[cell_i].size();
        for(size_t p=0;p<n_particles;p++)
        {
          const unsigned long long id = cells[cell_i][field::id][p];
          Vec3d pos = xform * Vec3d{cells[cell_i][field::rx][p],cells[cell_i][field::ry][p],cells[cell_i][field::rz][p]};
          if( prcsg.contains( pos , id ) )
          {
            cells[cell_i][field::id][p] = next_region_id.fetch_add(1,std::memory_order_relaxed);
          }
          else
          {
            cells[cell_i][field::id][p] = next_other_id.fetch_add(1,std::memory_order_relaxed);
          }
        }
      }

      ParticleRegion r = {};
      r.set_name( *name );
      r.m_id_start = 0;      
      r.m_id_end = total_particles_in_region;
      r.m_id_range_flag = true;

      ldbg << "add tracking region "<<r.name()<<" with "<< (r.m_id_end - r.m_id_start) <<" particles starting at id="<<r.m_id_start<<std::endl;

      particle_regions->push_back( r );
    }

  };

  template<class GridT> using TrackRegionParticlesTmpl = TrackRegionParticles< GridT >;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "track_region_particles", make_grid_variant_operator<TrackRegionParticlesTmpl> );
  }

}

