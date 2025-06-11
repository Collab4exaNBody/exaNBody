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
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/domain.h>
#include <exanb/core/particle_type_id.h>

#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>

#include <exanb/core/check_particles_inside_cell.h>
#include <onika/physics/constants.h>
#include <onika/thread.h>

#include <mpi.h>
#include <string>

namespace exanb
{
  struct ParticleInitVec
  {
    std::vector< onika::math::Vec3d > m_positions;
    std::vector< onika::math::Vec3d > m_velocities;
    std::vector< std::string > m_types;
  };
}

namespace YAML
{
  template<> struct convert< exanb::ParticleInitVec >
  {
    static bool decode(const Node& node, exanb::ParticleInitVec & v)
    {
      v.m_positions.clear();
      v.m_velocities.clear();
      v.m_types.clear();
      
      if( !node.IsSequence() )
      {
        onika::lerr << "ParticleInitVec expects a list" << std::endl;
        return false;
      }
      const size_t sz = node.size();
      v.m_positions.assign( sz , onika::math::Vec3d{0.,0.,0.} );
      v.m_velocities.assign( sz , onika::math::Vec3d{0.,0.,0.} );
      v.m_types.assign( sz , "0" );
      for(size_t i=0;i<sz;i++)
      {
        if( ! node[i].IsMap() ) { onika::lerr << "element #"<<i<<" in particle initialization list ist not a map"<<std::endl; return false; }
        if(node[i]["pos"]) { v.m_positions[i] = node[i]["pos"].as< onika::math::Vec3d >(); }
        else { onika::lerr << "element #"<<i<<" does not have a 'pos' entry"<<std::endl; return false; }
        if( node[i]["vel"] ) v.m_velocities[i] = node[i]["vel"].as< onika::math::Vec3d >();
        if( node[i]["type"] ) v.m_types[i] = node[i]["type"].as< std::string >();
      }
      return true;
    }
  };
}

namespace exanb
{

  template< class GridT >
  class GridInsertParticles : public OperatorNode
  {
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( MPI_Comm        , mpi          , INPUT , MPI_COMM_WORLD  );
    ADD_SLOT( Domain          , domain       , INPUT_OUTPUT );
    ADD_SLOT( GridT           , grid         , INPUT_OUTPUT );

    // get a type id from a type name
    ADD_SLOT( ParticleTypeMap , particle_type_map , INPUT , OPTIONAL );
    ADD_SLOT( ParticleInitVec , particles         , INPUT , REQUIRED );
    
  public:
    inline void execute () override final
    { 
      using has_field_type_t = typename GridT:: template HasField < field::_type >;
      static constexpr bool has_field_type = has_field_type_t::value;

      using has_field_id_t = typename GridT:: template HasField < field::_id >;
      static constexpr bool has_field_id = has_field_id_t::value;

      using ParticleTupleIO = std::conditional_t< has_field_type
                                                , onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_id, field::_type >
                                                , onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_id>
                                                >;

      const size_t n_particles = particles->m_positions.size();
      std::vector<int> type_indices;
      type_indices.assign( n_particles , 0 );
      for(size_t i=0;i<n_particles;i++)
      {
        const std::string& ts = particles->m_types[i];
        int t = -1;
        if( particle_type_map.has_value() )
        {
          auto it = particle_type_map->find( ts );
          if( it != particle_type_map->end() ) t = it->second;
        }
        if( t == -1 )
        {
          try
          {
              t = std::stoi(ts);
          }
          catch (std::invalid_argument const& ex)
          {
            fatal_error()<<"particle type '"<<ts<<"' is invalid" << std::endl;
          }
        }
        type_indices[i] = t;
      }

      const size_t n_cells = grid->number_of_cells();
      spin_mutex_array cell_locks;
      cell_locks.resize( n_cells );
      auto cells = grid->cells();
      const IJK local_grid_dim = grid->dimension();
      
      // find highest existing particle id
      unsigned long long next_id = 0;
      if( has_field_id )
      {
#       pragma omp parallel for schedule(dynamic) reduction(max:next_id)
        for(size_t cell_i=0;cell_i<n_cells;cell_i++) if( ! grid->is_ghost_cell(cell_i) )
        {
          size_t n_particles = cells[cell_i].size();
          for(size_t p=0;p<n_particles;p++)
          {
            const unsigned long long id = cells[cell_i][field::id][p];
            next_id = std::max( next_id , id );
          }
        }
        MPI_Allreduce(MPI_IN_PLACE,&next_id,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
        ++ next_id; // start right after greatest id
      }

      // insert particles into existing grid
      const uint64_t no_id = next_id;
      const Mat3d inv_xform = domain->inv_xform();
      unsigned long long local_generated_count = 0;
      
#     pragma omp parallel for schedule(dynamic) reduction(+:local_generated_count)
      for(size_t i=0;i<n_particles;i++)
      {
        Vec3d lab_pos = particles->m_positions[i];
        Vec3d grid_pos = inv_xform * lab_pos;
        const IJK loc = grid->locate_cell(grid_pos);
        if( grid->contains(loc) && is_inside( domain->bounds() , grid_pos ) && is_inside( grid->grid_bounds() , grid_pos ) )
        {          			
          assert( min_distance_between( grid_pos, grid->cell_bounds(loc) ) <= grid->cell_size()/2.0 );
          size_t cell_i = grid_ijk_to_index( local_grid_dim, loc);

          const auto v = particles->m_velocities[i];
          ParticleTupleIO pt;
          if constexpr (  has_field_type ) pt = ParticleTupleIO( grid_pos.x,grid_pos.y,grid_pos.z, v.x, v.y,v.z, no_id, type_indices[i] );
          if constexpr ( !has_field_type ) pt = ParticleTupleIO( grid_pos.x,grid_pos.y,grid_pos.z, v.x, v.y,v.z, no_id );
          
          cell_locks[cell_i].lock();
          cells[cell_i].push_back( pt , grid->cell_allocator() );
          cell_locks[cell_i].unlock();

          ++ local_generated_count;
        }
      }
      
      if constexpr ( has_field_id )
      {
        unsigned long long particle_id_start = 0;
        MPI_Exscan( &local_generated_count , &particle_id_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi);
        std::atomic<uint64_t> particle_id_counter = particle_id_start + next_id;

        const IJK dims = grid->dimension();
        const ssize_t gl = grid->ghost_layers();
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
      
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(grid_insert_particles)
  {
    OperatorNodeFactory::instance()->register_factory("grid_insert_particles", make_grid_variant_operator< GridInsertParticles >);
  }


}
