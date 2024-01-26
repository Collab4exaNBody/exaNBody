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

#include <onika/cuda/cuda.h>

#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/grid_cell_particles/particle_cell_projection.h>
#include <exanb/core/xform.h>

namespace exanb
{
  // enables selection of certain particles in space,
  // with the help for some discretized mask and/or geometrical region CSG
  template<class GridT, class XFormT>
  struct PartcileLocalizedFilter
  {
    GridT& grid;
    XFormT xform;
    ParticleRegionCSGShallowCopy prcsg;
    const double * __restrict__ cell_value_ptr = nullptr;
    size_t cell_value_stride = 0;
    double mask_value = 0.0;
    unsigned int subdiv = 0;

    inline void initialize_from_optional_parameters(
                  const ParticleRegions* particle_regions, 
                  ParticleRegionCSG* region,
                  const GridCellValues* grid_cell_values = nullptr, 
                  const std::string* grid_cell_mask_name = nullptr, 
                  const double* grid_cell_mask_value = nullptr )
    {
      // filter based on region CSG
      prcsg = ParticleRegionCSGShallowCopy{};
      if( region != nullptr )
      {
        if( particle_regions == nullptr ) { fatal_error() << "region is defined, but particle_regions has no value" << std::endl; }
        if( region->m_nb_operands==0 ) { region->build_from_expression_string( particle_regions->data() , particle_regions->size() ); }
        prcsg = *region;
      }        

      // filter based on masked locations
      cell_value_ptr = nullptr;
      cell_value_stride = 0;
      mask_value = 0.0;
      subdiv = 0;
      if( grid_cell_mask_name != nullptr )
      {
        //const double cell_size = grid.cell_size();
        //double subcell_size = 0.0; //cell_size / m_subdiv;

        if( grid_cell_values == nullptr )
        {
          fatal_error() << "cell mask specified but grid_cell_values has no value" << std::endl;          
        }
        if( grid_cell_mask_value == nullptr )
        {
          fatal_error() << "cell mask specified but grid_cell_mask_value has no value" << std::endl;          
        }
        mask_value = *grid_cell_mask_value;
        const auto accessor = grid_cell_values->field_data( *grid_cell_mask_name );
        const auto info = grid_cell_values->field(*grid_cell_mask_name);
        cell_value_ptr = accessor.m_data_ptr;
        cell_value_stride = accessor.m_stride;
        subdiv = info.m_subdiv;
        //subcell_size = cell_size / subdiv;
        if( subdiv*subdiv*subdiv != info.m_components )
        {
          fatal_error() << "expected a scalar value field for cell mask" << std::endl;          
        }
      }
    }

    ONIKA_HOST_DEVICE_FUNC inline bool operator () ( const Vec3d& r , uint64_t id ) const
    {
      using namespace ParticleCellProjectionTools;
      
      bool keep_particle = prcsg.contains(xform.transformCoord(r),id);
      if( keep_particle && cell_value_ptr!=nullptr )
      {
        const IJK dims = grid.dimension();
        const IJK loc = grid.locate_cell(r);
        const Vec3d cell_origin = grid.cell_position( loc );
        IJK center_cell_loc;
        IJK center_subcell_loc;
        Vec3d rco = r - cell_origin;
        const double cell_size = grid.cell_size();
        const double subcell_size = cell_size / subdiv;
        localize_subcell( rco, cell_size, subcell_size, subdiv, center_cell_loc, center_subcell_loc );
        center_cell_loc += loc;

        const ssize_t nbh_cell_i = grid_ijk_to_index( dims , center_cell_loc );
        const ssize_t nbh_subcell_i = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , center_subcell_loc );
        assert( nbh_cell_i>=0 && nbh_cell_i<ssize_t(grid.number_of_cells()) );
        assert( nbh_subcell_i>=0 && nbh_subcell_i<(subdiv*subdiv*subdiv) );
        auto cell_value = cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i ];
        keep_particle = ( cell_value == mask_value );  
      }
      return keep_particle;
    }
  };

}


