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
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <onika/parallel/random.h>
#include <onika/thread.h>
#include <exanb/core/particle_type_id.h>
#include <exanb/grid_cell_particles/particle_localized_filter.h>
#include <exanb/grid_cell_particles/lattice_collection.h>

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
    , LatticeCollection lattice  
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

    Vec3d lattice_size = lattice.m_size;

    // MPI Initialization
    int rank=0, np=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    lout << std::defaultfloat
         << "======= Lattice generator ======="<< std::endl
         << "structure         = " << lattice.m_structure << std::endl
         << "types             =";
    for(const auto& s:lattice.m_types) lout <<" "<<s;
    lout << std::endl
         << "lattice cell size = "<< lattice_size << std::endl
         << "position shift    = "<< position_shift <<std::endl;

    ParticleTypeMap typeMap;
    if( ! particle_type_map.empty() )
    {
      typeMap = particle_type_map;
    }
    else
    {
      for(const auto & type_name : lattice.m_types)
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
    size_t n_particles_cell = lattice.m_np;
    std::vector<unsigned int> particle_type_id(n_particles_cell);
    for(size_t i=0;i<n_particles_cell;i++) { particle_type_id[i] = typeMap.at( lattice.m_types.at(i) ); }
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
    Vec3d lab_width = lab_hi-lab_lo;
    
    Vec3d motif_lo = Vec3d{0., 0., 0.};
    Vec3d motif_hi = Vec3d{lattice_size.x, lattice_size.y, lattice_size.z};
    Vec3d motif_width = motif_hi-motif_lo;

    Vec3d rot_motif_lo = lattice.m_rotmat * motif_lo;
    Vec3d rot_motif_hi = lattice.m_rotmat * motif_hi;
    
    Vec3d corner0 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{            0.,             0.,             0.};
    Vec3d corner1 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{rot_motif_hi.x,             0.,             0.};
    Vec3d corner2 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{            0., rot_motif_hi.y,             0.};
    Vec3d corner3 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{            0.,             0., rot_motif_hi.z};
    Vec3d corner4 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{rot_motif_hi.x, rot_motif_hi.y,             0.};
    Vec3d corner5 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{rot_motif_hi.x,             0., rot_motif_hi.z};
    Vec3d corner6 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{            0., rot_motif_hi.y, rot_motif_hi.z};
    Vec3d corner7 = Vec3d{rot_motif_lo.x, rot_motif_lo.y, rot_motif_lo.z} + Vec3d{rot_motif_hi.x, rot_motif_hi.y, rot_motif_hi.z};        

    
    double xmin = std::min({corner0.x,corner1.x,corner2.x,corner3.x,corner4.x,corner5.x,corner6.x,corner7.x});
    double xmax = std::max({corner0.x,corner1.x,corner2.x,corner3.x,corner4.x,corner5.x,corner6.x,corner7.x});
    
    double ymin = std::min({corner0.y,corner1.y,corner2.y,corner3.y,corner4.y,corner5.y,corner6.y,corner7.y});    
    double ymax = std::max({corner0.y,corner1.y,corner2.y,corner3.y,corner4.y,corner5.y,corner6.y,corner7.y});
    
    double zmin = std::min({corner0.z,corner1.z,corner2.z,corner3.z,corner4.z,corner5.z,corner6.z,corner7.z});
    double zmax = std::max({corner0.z,corner1.z,corner2.z,corner3.z,corner4.z,corner5.z,corner6.z,corner7.z});
    
    AABB new_motif_bounds  = { { xmin, ymin, zmin } , { xmax, ymax, zmax } };
    Vec3d new_motif_width  = Vec3d{xmax-xmin, ymax-ymin, zmax-zmin};
    
    std::cout << "lab_lo = " << lab_lo << std::endl;
    std::cout << "lab_hi = " << lab_hi << std::endl;
    IJK lattice_lo = { static_cast<ssize_t>( (lab_lo.x-xmin) / new_motif_width.x )
      , static_cast<ssize_t>( (lab_lo.y-ymin) / new_motif_width.y )
      , static_cast<ssize_t>( (lab_lo.z-zmin) / new_motif_width.z ) };
    IJK lattice_hi = { static_cast<ssize_t>( (lab_hi.x-xmax) / new_motif_width.x )
      , static_cast<ssize_t>( (lab_hi.y-ymax) / new_motif_width.y )
      , static_cast<ssize_t>( (lab_hi.z-zmax) / new_motif_width.z ) };
    ssize_t i_start = lattice_lo.i - 1;
    ssize_t i_end   = lattice_hi.i + 1;
    ssize_t j_start = lattice_lo.j - 1;
    ssize_t j_end   = lattice_hi.j + 1;
    ssize_t k_start = lattice_lo.k - 1;
    ssize_t k_end   = lattice_hi.k + 1;

#   pragma omp parallel
    {
#     pragma omp for collapse(3) reduction(+:local_generated_count)
      for (ssize_t k=k_start; k<=k_end; k++)
      {
        for (ssize_t j=j_start; j<=j_end; j++)
        {
          for (ssize_t i=i_start; i<=i_end; i++)
      		{
	          for (size_t l=0; l<n_particles_cell;l++)
    		    {
              Vec3d lab_pos = ( Vec3d{ i + lattice.m_positions[l].x , j + lattice.m_positions[l].y , k + lattice.m_positions[l].z } * lattice.m_size );// + position_shift;
              Mat3d rot = lattice.m_rotmat;
              Vec3d lab_pos_rot = rot * lab_pos;
              
              Vec3d grid_pos = inv_xform * lab_pos_rot;
	            const IJK loc = grid.locate_cell(grid_pos); //domain_periodic_location( domain , pos );

	            if( grid.contains(loc) && is_inside( domain.bounds() , grid_pos ) && is_inside( grid.grid_bounds() , grid_pos ) )
        			{          			
                grid_pos = inv_xform * lab_pos_rot;

                if( particle_filter(grid_pos,no_id) && live_or_die_void_porosity(lab_pos_rot,void_centers,void_radiuses) )
	              {
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

#     pragma omp parallel
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
