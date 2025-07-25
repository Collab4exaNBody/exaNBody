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
#include <exanb/core/domain.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>

#include <exanb/mesh/triangle_mesh.h>
#include <exanb/mesh/triangle_math.h>
#include <exanb/mesh/edge_math.h>
#include <exanb/mesh/particle_mesh_proximty.h>

#include <onika/math/basic_types.h>

namespace exanb
{
  
  class GridTriangleIntersections : public OperatorNode
  {
    ADD_SLOT( onika::math::AABB            , bounding_box      , INPUT , REQUIRED );
    ADD_SLOT( long                         , resolution        , INPUT , REQUIRED , 128 );
    ADD_SLOT( double                       , distance          , INPUT , REQUIRED , 0.0 );
    ADD_SLOT( TriangleMesh                 , mesh              , INPUT , TriangleMesh{} );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT_OUTPUT );
    ADD_SLOT( bool                         , verify_result     , INPUT , false );

  public:
    inline bool is_sink() const override final
    {
      return true;
    }

    inline void execute() override final
    {
      const auto max_dist = *distance;
      
      auto grid_size = bounds_size( *bounding_box );
      const double length = std::max( std::max( grid_size.x , grid_size.y ) , grid_size.z );
      const double cell_size = length / (*resolution);
      const IJK dims = { static_cast<ssize_t>(ceil(grid_size.x/cell_size))
                       , static_cast<ssize_t>(ceil(grid_size.y/cell_size))
                       , static_cast<ssize_t>(ceil(grid_size.z/cell_size)) };
      grid_size = dims * cell_size;
      
      const double max_dist_cells = max_dist / cell_size;
      const auto grid_origin = bounding_box->bmin;
      
      grid_to_triangles->m_grid_dims = dims;
      grid_to_triangles->m_cell_size = cell_size;
      grid_to_triangles->m_origin = grid_origin;
      //grid_to_triangles->m_max_dist = max_dist;      
      
      ldbg << "grid_origin = " << grid_origin << std::endl;
      ldbg << "grid_size = " << grid_size << std::endl;
      ldbg << "cell_size = " << cell_size << std::endl;
      
      const size_t n_triangles = mesh->triangle_count();
      const size_t n_cells = dims.i * dims.j * dims.k;

      ldbg << "n_cells = " << n_cells << std::endl;
      ldbg << "n_triangles = " << n_triangles << std::endl;

      std::vector< std::vector<size_t> > cell_tri_indices( n_cells );
      size_t total_tri_indices = 0;

      for(size_t t=0;t<n_triangles;t++)
      {
        auto tri = mesh->triangle(t);
        for(int p=0;p<3;p++) tri[p] = ( tri[p] - grid_origin ) / cell_size;
        const auto tri_bounds = bounds( tri );
        const IJK start = { std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.x-max_dist_cells)))
                          , std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.y-max_dist_cells))) 
                          , std::max(ssize_t(0),static_cast<ssize_t>(floor(tri_bounds.bmin.z-max_dist_cells))) };
        const IJK end = { std::min(dims.i,static_cast<ssize_t>(ceil(tri_bounds.bmax.x+max_dist_cells))) 
                        , std::min(dims.j,static_cast<ssize_t>(ceil(tri_bounds.bmax.y+max_dist_cells))) 
                        , std::min(dims.k,static_cast<ssize_t>(ceil(tri_bounds.bmax.z+max_dist_cells))) };
        for(int k=start.k;k<end.k;k++)
        for(int j=start.j;j<end.j;j++)
        for(int i=start.i;i<end.i;i++)
        {
          const onika::math::AABB box = { { i-max_dist_cells , j-max_dist_cells , k-max_dist_cells } , { i+1+max_dist_cells , j+1+max_dist_cells , k+1+max_dist_cells } };
          if( intersect( box , tri ) )
          {
            const auto cell_idx = grid_ijk_to_index( dims , {i,j,k} );
            assert( cell_idx >= 0 && cell_idx < ssize_t(n_cells) );
            cell_tri_indices[cell_idx].push_back(t);
            ++ total_tri_indices;
          }
        }
      }
      
      ldbg << "triangle-cell insertions = "<<total_tri_indices << std::endl;
      
      auto & cell_triangles = grid_to_triangles->m_cell_triangles;      
      cell_triangles.clear();
      cell_triangles.assign( n_cells + 1 + total_tri_indices , 0 );
      size_t next_tri_idx = n_cells + 1;
      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        cell_triangles[cell_idx] = next_tri_idx;
        for(const auto t : cell_tri_indices[cell_idx])
        {
          cell_triangles[ next_tri_idx ++ ] = t;
        }
      }
      assert( next_tri_idx == ( n_cells + 1 + total_tri_indices ) );
      cell_triangles[n_cells] = next_tri_idx;


      if( *verify_result )
      {        
        auto cell_triangles_ro = read_only_view( *grid_to_triangles );

        // 1) Test box/triangle intersections with acceleration structure and with brute force triangle scan,
        // then compares the 2 results.
        {
          std::set<size_t> tla;
          std::set<size_t> tlb;
          for(int k=0;k<dims.k;k++)
          for(int j=0;j<dims.j;j++)
          for(int i=0;i<dims.i;i++)
          {
            const onika::math::AABB box = { grid_origin + Vec3d{i+0.,j+0.,k+0.} * cell_size - max_dist
                                          , grid_origin + Vec3d{i+1.,j+1.,k+1.} * cell_size + max_dist };
            const auto cell_idx = grid_ijk_to_index( dims , {i,j,k} );
            for(const auto t : cell_triangles_ro.cell_triangles(cell_idx))
            {
              if( intersect( box , mesh->triangle(t) ) ) tla.insert( t );
            }
            for(size_t t=0;t<n_triangles;t++)
            {
              if( intersect( box , mesh->triangle(t) ) ) tlb.insert( t );
            }
            if( tla != tlb )
            {
              fatal_error() << "cell #"<<cell_idx<<" @"<<IJK{i,j,k}<<" : triangle list mismatch, tla="<<tla.size()<<" , tlb="<<tlb.size()<<std::endl;
            }
          }
        }
        
        // 2) Test particle/triangle proximity with both acceleration structure and brute force triangle scan
        {
          std::set< onika::oarray_t<ssize_t,4> > set_a;
          auto check_particle_triangle_a = [&set_a](const Vec3d& r, const Vec3d& pv, double d, size_t /*unused*/, size_t p_idx, ssize_t v0, ssize_t v1, ssize_t v2, double w0, double w1, double w2, bool behind ) -> void
          {
            set_a.insert( { ssize_t(p_idx) , v0 , v1 , v2 } );
          };
          TrivialTriangleLocator all_triangles = { { 0 , mesh->triangle_count() } };
          ParticleTriangleProjProximity<decltype(check_particle_triangle_a),true,true> triv_proximity = { all_triangles , read_only_view(*mesh) , max_dist , check_particle_triangle_a };

          std::set< onika::oarray_t<ssize_t,4> > set_b;
          auto check_particle_triangle_b = [&set_b](const Vec3d& r, const Vec3d& pv, double d, size_t /*unused*/, size_t p_idx, ssize_t v0, ssize_t v1, ssize_t v2, double w0, double w1, double w2, bool behind ) -> void
          {
            set_b.insert( { ssize_t(p_idx) , v0 , v1 , v2 } );
          };
          GridParticleTriangleProjProximity<decltype(check_particle_triangle_b),true,true> grid_proximity = { cell_triangles_ro , read_only_view(*mesh) , max_dist , check_particle_triangle_b };

          const auto check_domain = enlarge( *bounding_box , cell_size * 2 );
          constexpr size_t N_SAMPLES = 10000;
          std::random_device rd;  //Will be used to obtain a seed for the random number engine
          std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
          std::uniform_real_distribution<double> rnd_x( check_domain.bmin.x , check_domain.bmax.x );
          std::uniform_real_distribution<double> rnd_y( check_domain.bmin.y , check_domain.bmax.y );
          std::uniform_real_distribution<double> rnd_z( check_domain.bmin.z , check_domain.bmax.z );
          for(size_t i=0;i<N_SAMPLES;i++)
          {
            const auto rx = rnd_x(gen);
            const auto ry = rnd_y(gen);
            const auto rz = rnd_z(gen);
            triv_proximity( 0 , i , rx, ry, rz );
            grid_proximity( 0 , i , rx, ry, rz );
          }
          if( set_a != set_b )
          {
            fatal_error() << "triangle / particle lists mismatch , set_a="<<set_a.size()<<" , set_b="<<set_b.size()<<std::endl;
          }
        } // end of test #2

      }

    }
  };

   // === register factories ===  
  ONIKA_AUTORUN_INIT(grid_triangle_intersections)
  {
    OperatorNodeFactory::instance()->register_factory("grid_triangle_intersections",make_simple_operator< GridTriangleIntersections >);
  }

}
