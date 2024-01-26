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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>

namespace exanb
{

  template< class GridT >
  class SetCellValues : public OperatorNode
  {  
    using DoubleVector = std::vector<double>;
  
    ADD_SLOT( GridT            , grid             , INPUT , REQUIRED );
    ADD_SLOT( Domain           , domain           , INPUT , REQUIRED );
    
    ADD_SLOT( ParticleRegions  , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG, region           , INPUT_OUTPUT , OPTIONAL );

    ADD_SLOT( GridCellValues   , grid_cell_values , INPUT_OUTPUT );
    ADD_SLOT( long             , grid_subdiv      , INPUT , 1 );
    ADD_SLOT( std::string      , field_name       , INPUT , REQUIRED );
    ADD_SLOT( DoubleVector     , value            , INPUT , DoubleVector{0.0} );

  public:  
    inline void execute() override final
    {
      // initialization of localization based particle filter (regions and masking)
      ParticleRegionCSGShallowCopy prcsg = {};
      if( region.has_value() )
      {
        if( region->m_nb_operands==0 )
        {
          region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
        }
      }
      prcsg = *region;

      if( value->empty() )
      {
        fatal_error() << "Cannot initialize cell values with 0-component vector value (empty vector given)" << std::endl;
      }

      if( grid->dimension() != grid_cell_values->grid_dims() )
      {
        ldbg << "Update cell values grid dimension to "<< grid->dimension() << " , existing values are discarded" << std::endl;
        grid_cell_values->set_grid_dims( grid->dimension() );
      }

      if( grid->offset() != grid_cell_values->grid_offset() )
      {
        ldbg << "Update cell values grid offset to "<< grid->offset() << std::endl;
        grid_cell_values->set_grid_offset( grid->offset() );
      }

      // retreive field data accessor. create data field if needed
      const int ncomps = value->size();
      const int subdiv = *grid_subdiv;
      if( ! grid_cell_values->has_field(*field_name) )
      {
        ldbg << "Create cell field "<< *field_name << " subdiv="<<subdiv<<" ncomps="<<ncomps<< std::endl;
        ldbg << "init value =";
        for(auto x:*value) ldbg << " "<<x;
        ldbg << std::endl;
        grid_cell_values->add_field(*field_name,subdiv,ncomps);
      }
      assert( size_t(subdiv) == grid_cell_values->field(*field_name).m_subdiv );
      assert( size_t(subdiv * subdiv * subdiv) * ncomps == grid_cell_values->field(*field_name).m_components );
      auto field_data = grid_cell_values->field_data(*field_name);
      
      const Mat3d xform = domain->xform();
      const double cell_size = domain->cell_size();
      const double subcell_size = cell_size / subdiv;
      const IJK dims = grid_cell_values->grid_dims();
      const IJK grid_offset = grid_cell_values->grid_offset();
      const Vec3d domain_origin = domain->origin();

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,cell_i,cell_loc, schedule(static) )
        {
          const Vec3d cell_origin = domain_origin + ( ( cell_loc + grid_offset ) * cell_size );
          for(int ck=0;ck<subdiv;ck++)
          for(int cj=0;cj<subdiv;cj++)
          for(int ci=0;ci<subdiv;ci++)
          {
            const Vec3d scr = xform * ( cell_origin + Vec3d{ subcell_size * (ci+0.5) , subcell_size * (cj+0.5) , subcell_size * (ck+0.5) } );
            if( prcsg.contains(scr,0) )
            {
              const IJK sc { ci, cj, ck };
              const size_t j = cell_i * field_data.m_stride +  grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , sc ) * ncomps;
              for(int k=0;k<ncomps;k++)
              {
                field_data.m_data_ptr[j+k] = value->at(k);
              }
            }
          }
        }
        GRID_OMP_FOR_END
      }
      
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "set_cell_values", make_grid_variant_operator<SetCellValues> );
  }

}

