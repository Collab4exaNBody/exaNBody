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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
      
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace exanb
{

  template< class GridT >
  class ReadCellValues : public OperatorNode
  {  
    using DoubleVector = std::vector<double>;
  
    ADD_SLOT( GridT            , grid             , INPUT , REQUIRED );
    ADD_SLOT( Domain           , domain           , INPUT , REQUIRED );
    
    ADD_SLOT( ParticleRegions  , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG, region           , INPUT_OUTPUT , OPTIONAL , DocString{"Region of the field where the value is to be applied."} );
    ADD_SLOT( GridCellValues   , grid_cell_values , INPUT_OUTPUT );
    ADD_SLOT( long             , grid_subdiv      , INPUT , 1, DocString{"Number of (uniform) subdivisions required for this field. Note that the refinement is an octree."});    
    ADD_SLOT( std::string      , field_name       , INPUT , REQUIRED , DocString{"Name of the field."});
    ADD_SLOT( int              , field_dim       , INPUT , REQUIRED);    
    ADD_SLOT( std::string      , file_name        , INPUT , REQUIRED , DocString{"Name of the file."});

  public:  

    inline std::string documentation() const override final
    {
      return R"EOF(
              This operator assigns a value to a point on a Cartesian grid, such as the velocity of a fluid. This operator can also be used to refine the grid and define different behaviors in different spatial regions. 
                )EOF";
    }

    inline void execute() override final
    {

      if( file_name->empty() )
      {
        fatal_error() << "Cannot initialize cell values with 0-component vector value (empty vector given)" << std::endl;
      }

      if( *grid_subdiv > 1 )
      {
        fatal_error() << "Cannot assign subdiv > 1. The external field should match the exact size of the grid defined by domain" << std::endl;
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

      int Nx, Ny, Nz, dataDimension;
      std::string dataType;
      
      std::ifstream file(*file_name);
      if (!file.is_open()) {
        throw std::runtime_error(std::string("Could not open file ") + *file_name);
      }

      // retreive field data accessor. create data field if needed
      const int ncomps = *field_dim;
      const int subdiv = 1;
      if( ! grid_cell_values->has_field(*field_name) )
        {
          ldbg << "Create cell field "<< *field_name << " subdiv="<<subdiv<<" ncomps="<<ncomps<< std::endl;
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
      
      std::string line;
      bool dataSection = false;
      int pointCount = 0;
      size_t cnt = 0;
      while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;
        if (keyword == "DIMENSIONS") {
          lineStream >> Nx >> Ny >> Nz;
          if ( (dims.i != Nx ) || (dims.j != Ny ) || (dims.k != Nz ) ) {
            lerr << "VTK file grid dimensions are not equal to simulation grid dimensions" << std::endl;
            lerr << "Current  grid dimensions = " << grid->dimension() << std::endl;
            lerr << "VTK file grid dimensions = " << IJK{Nx,Ny,Nz} << std::endl;
            std::abort();
          }
        } else if (keyword == "POINT_DATA") {
          lineStream >> pointCount;
          assert(pointCount == Nx*Ny*Nz);
          assert((grid_dims.i*grid_dims.j*grid_dims.k) == pointCount);
        } else if (keyword == "SCALARS" || keyword == "VECTORS") {
          dataType = keyword;
          std::string name, type;
          lineStream >> name >> type;
          dataDimension = (keyword == "SCALARS") ? 1 : 3; // Scalars have 1 dimension, Vectors have 3
          cnt = 0;
        } else if (keyword == "LOOKUP_TABLE") {
          std::string lkp;
          lineStream >> lkp;
          dataSection = true;            
        }
        if (dataSection) {
          double value;
          while (lineStream >> value || std::getline(file, line)) {
            if (!lineStream) {
              lineStream.clear();
              lineStream.str(line);
              lineStream >> value;
            }

            for (size_t d = 0; d < dataDimension; ++d) {
              field_data.m_data_ptr[cnt+d] = value;
              if (d < dataDimension - 1) {
                lineStream >> value;
              }
            }
            cnt += 1;
          }
        }
      }

      file.close();
    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(read_cell_values)
  {
    OperatorNodeFactory::instance()->register_factory( "read_cell_values", make_grid_variant_operator<ReadCellValues> );
  }

}

