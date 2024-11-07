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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/parallel_random.h>
#include <exanb/core/thread.h>
#include <exanb/core/particle_type_id.h>
#include <exanb/grid_cell_particles/particle_localized_filter.h>

#include <mpi.h>
#include <string>
#include <numeric>
#include <memory>
#include <random>
#include <cmath>

namespace exanb
{
  // Generate Orthorhombic lattice, i.e. a, b, c may have different lengths but alpha, beta and gamma equal 90 degrees.
  // Requirements :
  // structure : SC, BCC, FCC, HCP, DIAMOND, ROCKSALT, FLUORITE, C15, PEROVSKITE, ST, BCT, FCT, 2BCT
  // types : number of types depends on the structure and types must be consistent with the species defined beforehand
  // size : vector containing a, b, c lengths of the lattice
  // repeats : number of repetition along three laboratory directions

  static inline bool live_or_die_void_porosity(const Vec3d& pos, const std::vector<Vec3d>& void_center, const std::vector<double>& void_radius)
  {
    for (size_t i=0; i < void_center.size(); i++)
    {
      if( distance(pos,void_center[i]) < void_radius[i] ) return false;
    }
    return true;
  }

  template<class GridT, class ParticleTypeField>
  static inline void generate_particle_lattice(
      MPI_Comm comm
    , ReadBoundsSelectionMode bounds_mode
    , const Domain& domain
    , GridT& grid
    , const ParticleTypeMap& particle_type_map
    , const ParticleRegions* particle_regions
    , ParticleRegionCSG* region
    , const GridCellValues* grid_cell_values
    , const std::string* grid_cell_mask_name
    , const double* grid_cell_mask_value
    , const ScalarSourceTermInstance user_function
    , double user_threshold
    , const std::string& structure
    , const std::vector<std::string>& types
    , double sigma_noise
    , const Vec3d& size
    , double noise_cutoff
    , const Vec3d& position_shift
    , const std::string& void_mode
    , const Vec3d& void_center
    , double void_radius
    , double void_porosity
    , double void_mean_diameter
    , ParticleTypeField )
  {
    using has_field_type_t = typename GridT:: template HasField < ParticleTypeField >;
    static constexpr bool has_field_type = has_field_type_t::value;

    using has_field_id_t = typename GridT:: template HasField < field::_id >;
    static constexpr bool has_field_id = has_field_id_t::value;

    using ParticleTupleIO = std::conditional_t< has_field_type
                                              , onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, ParticleTypeField>
                                              , onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id>
                                              >;

    Vec3d lattice_size = size;
    double noise_upper_bound = std::numeric_limits<double>::max();
    if( noise_cutoff >= 0.0 ) noise_upper_bound = noise_cutoff;
    else noise_upper_bound = std::min(lattice_size.x,std::min(lattice_size.y,lattice_size.z)) * 0.5;

    // MPI Initialization
    int rank=0, np=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    // uint64_t n_particles = 0;

    // Verification du nombre de types d'atomes en fonction du type de maille cubique voulu
    // ***choix**  **nb_types**
    //         SC   1  
    //        BCC   2
    //        FCC   4
    //        HCP   4
    //    DIAMOND   8
    //   ROCKSALT   8
    //   FLUORITE  12
    //        C15  24
    // PEROVSKITE   5
    //         ST   1
    //        BCT   2
    //        FCT   4
    //       2BCT   4

    lout << std::defaultfloat
         << "======= Lattice generator ======="<< std::endl
         << "structure         = " << structure << std::endl
         << "types             =";
    for(const auto& s:types) lout <<" "<<s;
    lout << std::endl
         << "lattice cell size = "<< lattice_size << std::endl
         << "position shift    = "<< position_shift <<std::endl
         << "noise sigma       = "<< sigma_noise <<std::endl
         << "noise cutoff      = "<< noise_cutoff <<std::endl;
    
    uint64_t n_particles_cell = 0.;
    std::vector<Vec3d> positions;
    if (structure == "SC" ) {
      n_particles_cell = 1;
      if( types.size() != n_particles_cell )
      {
        lerr << "Parameter types for simple cubic lattice must contain exactly 1 name" << std::endl;
        std::abort();
      }
      positions.resize(n_particles_cell);
      positions = {
        {.25, .25, .25} };
    } else if (structure == "BCC" ) {
      n_particles_cell = 2;
      if( types.size() != n_particles_cell )
      {
        lerr << "Parameter types for body centered cubic lattice must contain exactly 2 names" << std::endl;
        std::abort();
      }
      positions.resize(n_particles_cell);
      positions = {
        {.25, .25, .25} ,
        {.75, .75, .75} };          
    } else if (structure == "FCC" ) {
      n_particles_cell = 4;
      if( types.size() != n_particles_cell )
      {
        lerr << "Parameter types for face centered cubic lattice must contain exactly 4 names" << std::endl;
        std::abort();
      }
      positions.resize(n_particles_cell);
      positions = {
        {.25, .25, .25} ,
        {.25, .75, .75} ,
        {.75, .25, .75} ,
        {.75, .75, .25} };
    } else if (structure == "HCP" ) {
      n_particles_cell = 4;
      if( types.size() != n_particles_cell )
      {
        lerr << "Parameter types for hexagonal compact lattice must contain exactly 4 names" << std::endl;
        std::abort();
      }
      positions.resize(n_particles_cell);
      positions = {
        {0.25000000,    0.25000000,    0.25000000} ,
        {0.75000000,    0.75000000,    0.25000000} ,
        {0.25000000,    0.58333333,    0.75000000} ,
        {0.75000000,    0.08333333,    0.75000000} };

    	lattice_size.y = 2. * lattice_size.y * sin(120. * M_PI / 180.);
  	  ldbg << "hcp cell = " << lattice_size << std::endl;
    } else if (structure == "DIAMOND" ) {
      n_particles_cell = 8;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for diamond lattice must contain exactly 8 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.00, .00, .00} ,
        {.50, .50, .00} ,
        {.00, .50, .50} ,
        {.50, .00, .50} ,
        {.25, .25, .25} ,
        {.75, .75, .25} ,          
        {.75, .25, .75} ,
        {.25, .75, .75} };
    } else if (structure == "ROCKSALT" ) {
      n_particles_cell = 8;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for rocksalt lattice must contain exactly 8 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.00, .00, .00} ,
        {.50, .50, .00} ,
        {.00, .50, .50} ,
        {.50, .00, .50} ,
        {.50, .00, .00} ,
        {.00, .50, .00} ,          
        {.00, .00, .50} ,
        {.50, .50, .50} };
    } else if (structure == "FLUORITE" ) {
      n_particles_cell = 12;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for fluorite lattice must contain exactly 12 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.00, .00, .00} ,
        {.00, .50, .50} ,
        {.50, .00, .50} ,
        {.50, .50, .00} ,
        {.25, .25, .25} ,
        {.75, .25, .25} ,
        {.25, .75, .25} ,
        {.75, .75, .25} ,
        {.25, .25, .75} ,
        {.75, .25, .75} ,
        {.25, .75, .75} ,
        {.75, .75, .75} };
    } else if (structure == "C15" ) {
      n_particles_cell = 24;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for C15 (lava) lattice must contain exactly 24 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        // We consider here the Cu2Mg structure
        // Cu atoms
        {.50, .50, .50} ,
        {.50, .75, .75} ,
        {.75, .50, .75} ,
        {.75, .75, .50} ,
        {.25, .50, .25} ,
        {.50, .25, .25} ,
        {.25, .25, .50} ,
        {.25, .75, .00} ,
        {.50, .00, .00} ,
        {.25, .00, .75} ,
        {.00, .50, .00} ,
        {.75, .25, .00} ,
        {.00, .25, .75} ,
        {.00, .75, .25} ,
        {.75, .00, .25} ,
        {.00, .00, .50} ,
        // Mg atoms are at positions of the diamond lattice
        {.125, .125, .125} ,
        {.875, .875, .875} ,
        {.875, .375, .375} ,
        {.375, .875, .375} ,
        {.625, .125, .625} ,
        {.375, .375, .875} ,
        {.125, .625, .625} ,          
        {.625, .625, .125} };
    } else if (structure == "PEROVSKITE" ) {
      n_particles_cell = 5;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for perovskite lattice must contain exactly 5 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.50, .50, .50} ,
        {.00, .00, .00} ,
        {.50, .00, .00} ,
        {.00, .50, .00} ,
        {.00, .00, .50} };
    } else if (structure == "ST" ) {
      n_particles_cell = 1;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for simple tetragonal lattice must contain exactly 1 name" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.25, .25, .25} };
    } else if (structure == "BCT" ) {
      n_particles_cell = 2;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for body centered tetragonal  lattice must contain exactly 2 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
      {.25, .25, .25},
      {.75, .75, .75} };          
    } else if (structure == "FCT" ) {
      n_particles_cell = 4;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for face centered tetragonal lattice must contain exactly 4 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.25, .25, .25} ,
        {.25, .75, .75} ,
        {.75, .25, .75} ,
        {.75, .75, .25} };	
    }  else if (structure == "2BCT" ) {
      n_particles_cell = 4;
      if( types.size() != n_particles_cell )
        {
          lerr << "Parameter types for double body centered tetragonal lattice must contain exactly 4 names" << std::endl;
          std::abort();
        }
      positions.resize(n_particles_cell);
      positions = {
        {.00, .00, .00} ,
        {.00, .50, .25} ,
        {.50, .50, .50} ,
        {.50, .00, .75} };	
    }           

    ParticleTypeMap typeMap;
    if( ! particle_type_map.empty() )
    {
      typeMap = particle_type_map;
    }
    else
    {
      for(const auto & type_name : types)
      {
        const auto type_id = typeMap.size();
        typeMap[ type_name ] = type_id;
      }
    }

    // initialization of localization based particle filter (regions and masking)
    PartcileLocalizedFilter<GridT,LinearXForm> particle_filter = { grid, { domain.xform() } };
    particle_filter.initialize_from_optional_parameters( particle_regions,
                                                         region,
                                                         grid_cell_values, 
                                                         grid_cell_mask_name, 
                                                         grid_cell_mask_value,
                                                         user_function,
                                                         user_threshold );

    // Get max and min positions
    // Need to define the size of the box
    // NOTE : only one processor need to do that   
       
    const IJK local_grid_dim = grid.dimension();
    unsigned long long next_id = 0;      
    
    if( has_field_id )
    {
      size_t n_cells = grid.number_of_cells();
      auto cells = grid.cells();
#       pragma omp parallel for schedule(dynamic) reduction(max:next_id)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++) if( ! grid.is_ghost_cell(cell_i) )
      {
        size_t n_particles = cells[cell_i].size();
        for(size_t p=0;p<n_particles;p++)
        {
          const unsigned long long id = cells[cell_i][field::id][p];
          next_id = std::max( next_id , id );
        }
      }
      MPI_Allreduce(MPI_IN_PLACE,&next_id,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
      ++ next_id; // start right after gretest id
    }

    // =============== Section that concerns the porosity mode ========================

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_radius(void_mean_diameter, void_mean_diameter/2.);
    std::uniform_real_distribution<double> distribution_center_x( domain.origin().x , domain.extent().x );
    std::uniform_real_distribution<double> distribution_center_y( domain.origin().y , domain.extent().y );
    std::uniform_real_distribution<double> distribution_center_z( domain.origin().z , domain.extent().z );      
    
    std::vector<Vec3d> void_centers;
    std::vector<double> void_radiuses;
    
    if( void_mode == "porosity" )
    {
      double calc_porosity = 0.;
      double void_volume = 0.;
      double volume = 1.0;
      const auto a = column1( domain.xform() ); // m11, domain.xform().m21, domain.xform().m31 };
      const auto b = column2( domain.xform() ); // { domain.xform().m12, domain.xform().m22, domain.xform().m32 };
      const auto c = column3( domain.xform() ); // { domain.xform().m13, domain.xform().m23, domain.xform().m33 };
      volume = dot( cross(a,b) , c );
      volume *= bounds_volume( domain.bounds() );          
      while ( calc_porosity < void_porosity)
      {
        double x = distribution_center_x(generator);
        double y = distribution_center_y(generator);
        double z = distribution_center_z(generator);            
        void_centers.push_back( {x, y, z} );
        void_radiuses.push_back( distribution_radius(generator) );
        void_volume += 4.*M_PI*pow(distribution_radius(generator),3.)/3.;
        calc_porosity = void_volume/volume;
      }
    }
    else if( void_mode == "simple" )
    {
      void_centers.push_back( void_center );
      void_radiuses.push_back( void_radius );
    }
    else if( void_mode == "none" )
    {
      // nothing to do
    }
    else
    {
      fatal_error() << "void_mode '"<< void_mode << "' unknown" << std::endl;
    }

    // =============== Generate particles into grid ========================

    //std::vector<unsigned long> particle_type_count( n_particles_cell , 0 );
    std::vector<unsigned int> particle_type_id(n_particles_cell);
    for(size_t i=0;i<n_particles_cell;i++) { particle_type_id[i] = typeMap.at( types.at(i) ); }
    if( typeMap.size()>1 && !has_field_type )
    {
      lerr<<"Warning: particle type is ignored"<<std::endl;
    }
    
    spin_mutex_array cell_locks;
    cell_locks.resize( grid.number_of_cells() );
    auto cells = grid.cells();

    //ldbg << "total particles = "<< repeats->i * repeats->j * repeats->k * n_particles_cell<<std::endl;

    const uint64_t no_id = next_id;
    unsigned long long local_generated_count = 0;

/*
    const size_t n_cells = grid.number_of_cells();
    std::vector<bool> empty_cell( n_cells , false );
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      empty_cell[cell_i] = cells[cell_i].empty();
    }    
*/

    const Mat3d inv_xform = domain.inv_xform();
    const AABB grid_bounds = grid.grid_bounds();
    Vec3d lab_lo = domain.xform() * grid_bounds.bmin;
    Vec3d lab_hi = domain.xform() * grid_bounds.bmax;
    IJK lattice_lo = { static_cast<ssize_t>( lab_lo.x / lattice_size.x )
                     , static_cast<ssize_t>( lab_lo.y / lattice_size.y )
                     , static_cast<ssize_t>( lab_lo.z / lattice_size.z ) };
    IJK lattice_hi = { static_cast<ssize_t>( lab_hi.x / lattice_size.x )
                     , static_cast<ssize_t>( lab_hi.y / lattice_size.y )
                     , static_cast<ssize_t>( lab_hi.z / lattice_size.z ) };
    ssize_t i_start = lattice_lo.i - 1;
    ssize_t i_end   = lattice_hi.i + 1;
    ssize_t j_start = lattice_lo.j - 1;
    ssize_t j_end   = lattice_hi.j + 1;
    ssize_t k_start = lattice_lo.k - 1;
    ssize_t k_end   = lattice_hi.k + 1;
    lout << "lattice start     = "<< i_start<<" , "<<j_start<<" , "<<k_start <<std::endl;
    lout << "lattice end       = "<< i_end<<" , "<<j_end<<" , "<<k_end <<std::endl;

#     pragma omp parallel
    {
      auto& re = rand::random_engine();
      std::normal_distribution<double> f_rand(0.,1.);
      
#       pragma omp for collapse(3) reduction(+:local_generated_count)
      for (ssize_t k=k_start; k<=k_end; k++)
      {
        for (ssize_t j=j_start; j<=j_end; j++)
        {
          for (ssize_t i=i_start; i<=i_end; i++)
      		{
	          for (size_t l=0; l<n_particles_cell;l++)
    		    {
              Vec3d lab_pos = ( Vec3d{ i + positions[l].x , j + positions[l].y , k + positions[l].z } * lattice_size ) + position_shift;
              Vec3d grid_pos = inv_xform * lab_pos;
	            const IJK loc = grid.locate_cell(grid_pos); //domain_periodic_location( domain , pos );

	            if( is_inside( domain.bounds() , grid_pos ) && is_inside( grid.grid_bounds() , grid_pos ) )
        			{          			
                Vec3d noise = Vec3d{ f_rand(re) * sigma_noise , f_rand(re) * sigma_noise , f_rand(re) * sigma_noise };
                const double noiselen = norm(noise);
                if( noiselen > noise_upper_bound ) noise *= noise_upper_bound/noiselen;
                lab_pos += noise;
                grid_pos = inv_xform * lab_pos;

                if( particle_filter(grid_pos,no_id) && live_or_die_void_porosity(lab_pos,void_centers,void_radiuses) )
	              {
	              	assert( grid.contains(loc) );
	              	assert( min_distance_between( grid_pos, grid.cell_bounds(loc) ) <= grid.cell_size()/2.0 );
	              	size_t cell_i = grid_ijk_to_index(local_grid_dim, loc);
                  ++ local_generated_count;
                  
                  ParticleTupleIO pt;
                  if constexpr (  has_field_type ) pt = ParticleTupleIO( grid_pos.x,grid_pos.y,grid_pos.z, no_id, particle_type_id[l] );
	              	if constexpr ( !has_field_type ) pt = ParticleTupleIO( grid_pos.x,grid_pos.y,grid_pos.z, no_id );
                  
	              	cell_locks[cell_i].lock();
	              	cells[cell_i].push_back( pt , grid.cell_allocator() );
	              	cell_locks[cell_i].unlock();
	              }
	            }
	          }
          }
        }
      }
      
    }

    grid.rebuild_particle_offsets();  

    ldbg << "MPI process generated "<< local_generated_count<<" particles"<<std::endl;

    if constexpr ( has_field_id )
    {
      unsigned long long particle_id_start = 0;
      MPI_Exscan( &local_generated_count , &particle_id_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , comm);
      std::atomic<uint64_t> particle_id_counter = particle_id_start + next_id;

      const IJK dims = grid.dimension();
      const ssize_t gl = grid.ghost_layers();
      const IJK gstart { gl, gl, gl };
      const IJK gend = dims - IJK{ gl, gl, gl };
      const IJK gdims = gend - gstart;

#       pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gstart );
          size_t n = cells[i].size();
          for(size_t j=0;j<n;j++)
          {
            if( cells[i][field::id][j] == no_id )
            {
              cells[i][field::id][j] = particle_id_counter.fetch_add(1,std::memory_order_relaxed);
            }
          }
        }
        GRID_OMP_FOR_END
      }
    }

    unsigned long long min_output_particles = 0;      
    unsigned long long max_output_particles = 0;      
    unsigned long long sum_output_particles = 0;      
    MPI_Allreduce(&local_generated_count,&min_output_particles,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);
    MPI_Allreduce(&local_generated_count,&max_output_particles,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
    MPI_Allreduce(&local_generated_count,&sum_output_particles,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    lout << "output particles  = " << sum_output_particles << " (min="<<min_output_particles<<",max="<<max_output_particles<<")" <<std::endl
         << "================================="<< std::endl << std::endl;
  }

}
