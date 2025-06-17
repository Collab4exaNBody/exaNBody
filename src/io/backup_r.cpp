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
#include <onika/math/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/backup_r.h>
#include <exanb/core/domain.h>

namespace exanb
{
/*
*/

  template<typename GridT>
  struct PositionBackupNode : public OperatorNode
  {
    ADD_SLOT( GridT        , grid     , INPUT );
    ADD_SLOT( Domain       , domain     , INPUT );
    ADD_SLOT( PositionBackupData , backup_r , INPUT_OUTPUT);

    inline void execute ()  override final
    {
      const IJK dims = grid->dimension();
      auto cells = grid->cells();
      const double cell_size = domain->cell_size();
      const ssize_t gl = grid->ghost_layers();

      const size_t n_cells = grid->number_of_cells();
      ldbg<<"PositionBackupNode : dims="<<dims<< " , n_cells=" << n_cells<< std::endl;

      backup_r->m_xform = domain->xform();
      backup_r->m_data.clear();
      backup_r->m_data.resize( grid->number_of_cells() );

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims-2*gl, _, loc_no_gl )
        {
          const IJK loc = loc_no_gl + gl;
          const size_t i = grid_ijk_to_index( dims , loc );
	        const size_t n_particles = cells[i].size();
          backup_r->m_data[i].resize( n_particles * 3 );

	        const Vec3d cell_origin = grid->cell_position( loc );
          uint32_t* rb = backup_r->m_data[i].data();
          const auto* __restrict__ rx = cells[i][field::rx];
          const auto* __restrict__ ry = cells[i][field::ry];
          const auto* __restrict__ rz = cells[i][field::rz];

#         pragma omp simd
          for(size_t j=0;j<n_particles;j++)
          {
            rb[j*3+0] = encode_double_u32( rx[j] , cell_origin.x , cell_size );
            rb[j*3+1] = encode_double_u32( ry[j] , cell_origin.y , cell_size );
            rb[j*3+2] = encode_double_u32( rz[j] , cell_origin.z , cell_size );
          }
        }
        GRID_OMP_FOR_END
      }

/*
      double max_error = 0.0;
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc, reduction(max:max_error) )
        {
	        const size_t n_particles = cells[i].size();
	        assert( backup_r->m_data[i].size() == n_particles*3 );

	        const Vec3d cell_origin = grid->cell_position( loc );
          const uint32_t* rb = backup_r->m_data[i].data();
          const auto* __restrict__ rx = cells[i][field::rx];
          const auto* __restrict__ ry = cells[i][field::ry];
          const auto* __restrict__ rz = cells[i][field::rz];
          for(size_t j=0;j<n_particles;j++)
          {
            max_error = std::max( max_error , std::fabs( rx[j] - restore_u32_double(rb[j*3+0],cell_origin.x,cell_size) ) );
            max_error = std::max( max_error , std::fabs( ry[j] - restore_u32_double(rb[j*3+1],cell_origin.y,cell_size) ) );
            max_error = std::max( max_error , std::fabs( rz[j] - restore_u32_double(rb[j*3+2],cell_origin.z,cell_size) ) );
          }
        }
        GRID_OMP_FOR_END
      }
      std::cout << "max_error = "<<max_error<<std::endl;
*/
    }


  };
    
 // === register factories ===  
  ONIKA_AUTORUN_INIT(backup_r)
  {
   OperatorNodeFactory::instance()->register_factory( "backup_r", make_grid_variant_operator< PositionBackupNode > );
  }

}

