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
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>

#include <onika/cuda/cuda_context.h>
#include <onika/memory/allocator.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>

#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_scratch.h>
#include <exanb/particle_neighbors/chunk_neighbors_host_write_accessor.h>

#include <exanb/core/domain.h>
#include <exanb/core/xform.h>

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

#include <exanb/mesh/triangle_mesh.h>
#include <exanb/mesh/triangle_math.h>
#include <exanb/mesh/particle_mesh_collision.h>
#include <exanb/mesh/edge_math.h>


namespace exanb
{

  template<class GridT, class CellsT, class GridTriangleLocatorT>
  struct NoSurfCrossingNeighborFilter
  {
    const GridT& m_grid;
    CellsT m_cells;
    GridTriangleLocatorT m_collision_func;
    bool m_half_symmetric = false;
    bool m_skip_ghost = false;
    
    inline bool operator () (double d2, double rcut2, size_t cell_a, size_t p_a, size_t cell_b, size_t p_b) const
    {
      assert( cell_a!=cell_b || p_a!=p_b );
      if( d2 > 0.0 && d2 <= rcut2 )
      {
        if( m_half_symmetric ) { if( cell_a<cell_b || ( cell_a==cell_b && p_a<p_b ) ) return false; }
        if( m_skip_ghost     ) { if( m_grid.is_ghost_cell(cell_b)                   ) return false; }
        const double rx = m_cells[cell_a][field::rx][p_a];
        const double ry = m_cells[cell_a][field::ry][p_a];
        const double rz = m_cells[cell_a][field::rz][p_a];
        const double dx = m_cells[cell_b][field::rx][p_b] - rx;
        const double dy = m_cells[cell_b][field::ry][p_b] - ry;
        const double dz = m_cells[cell_b][field::rz][p_b] - rz;
        return ! m_collision_func( cell_a, p_a, rx,ry,rz, dx,dy,dz );
      }
      return false;
    }
  };

  template<typename GridT>
  struct ChunkNeighborsNoSurfaceCrossing : public OperatorNode
  {
    ADD_SLOT( GridT               , grid            , INPUT );
    ADD_SLOT( GridTriangleIntersectionList , grid_to_triangles , INPUT , OPTIONAL );
    ADD_SLOT( TriangleMesh                 , mesh              , INPUT, REQUIRED );
    ADD_SLOT( AmrGrid             , amr             , INPUT );
    ADD_SLOT( AmrSubCellPairCache , amr_grid_pairs  , INPUT );
    ADD_SLOT( Domain              , domain          , INPUT );
    ADD_SLOT( double              , nbh_dist_lab    , INPUT );
    ADD_SLOT( GridChunkNeighbors  , chunk_neighbors , INPUT_OUTPUT );

    ADD_SLOT( ChunkNeighborsConfig, config , INPUT_OUTPUT, ChunkNeighborsConfig{} );
    ADD_SLOT( ChunkNeighborsScratchStorage, chunk_neighbors_scratch, PRIVATE );

    inline void execute () override final
    {
      if( config->chunk_size != 1 )
      {
        lerr << "Warning, chunk size was "<< config->chunk_size<<", it is forced to 1"<<std::endl;
      }
      unsigned int cs = 1;
      unsigned int cs_log2 = 0;

      const bool gpu_enabled = (global_cuda_ctx() != nullptr) ? global_cuda_ctx()->has_devices() : false;
      static constexpr std::false_type no_z_order = {};
      
      auto cells = grid->cells_accessor();
      using CellsT = std::remove_reference_t< decltype(cells) >;

      LinearXForm xform = { domain->xform() };
      if( grid_to_triangles.has_value() )
      {
        GridParticleTriangleCollision<> func = { read_only_view(*grid_to_triangles) , read_only_view(*mesh) };
        NoSurfCrossingNeighborFilter<GridT,CellsT,GridParticleTriangleCollision<> > nbh_filter = { *grid, cells, func, config->half_symmetric , config->skip_ghosts };
        chunk_neighbors_execute(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter );        
      }
      else
      {
        TrivialTriangleLocator all_triangles = { { 0 , mesh->triangle_count() } };
        ParticleTriangleCollision<> func = { all_triangles , read_only_view(*mesh) };
        NoSurfCrossingNeighborFilter<GridT,CellsT,ParticleTriangleCollision<> > nbh_filter = { *grid, cells, func, config->half_symmetric , config->skip_ghosts };
        chunk_neighbors_execute(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter );        
      }

    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(chunk_neighbors_no_surface_crossing)
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_no_surface_crossing", make_grid_variant_operator< ChunkNeighborsNoSurfaceCrossing > );
  }

}

