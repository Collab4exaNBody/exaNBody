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
#include <onika/yaml/yaml_enum.h>
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

EXANB_YAML_ENUM(exanb,StructuredGridOrdering,C_ORDER,F_ORDER)

namespace exanb
{
  template< class GridT >
  class ReadCellValues : public OperatorNode
  {  
    using DoubleVector = std::vector<double>;
  
    ADD_SLOT( Domain           , domain           , INPUT , REQUIRED );
    ADD_SLOT( GridT            , grid             , INPUT , REQUIRED );
    ADD_SLOT( GridCellValues   , grid_cell_values , INPUT_OUTPUT );
    ADD_SLOT( long             , grid_subdiv      , INPUT , DocString{"(YAML: int) Number of (uniform) subdivisions required for this field. Note that the refinement is an octree."});    
    ADD_SLOT( std::string      , field_name       , INPUT , REQUIRED , DocString{"(YAML: string) Name of the field."});
    ADD_SLOT( long             , field_dim        , INPUT , 1, DocString{"(YAML: int) Field dimension."});
    ADD_SLOT( std::string      , file_name        , INPUT , REQUIRED , DocString{"(YAML: string) Name of the file."});
    ADD_SLOT( StructuredGridOrdering , grid_ordering , INPUT , StructuredGridOrdering::F_ORDER , DocString{"(YAML: StructuredGridOrdering, i.e. C_ORDER or F_ORDER symbolic value) Ordering style."});

  public:  

    inline void execute() override final
    {

      if( *field_dim > 1 )
      {
        fatal_error() << "Cannot consider field_dim > 1. For now, the external field should be scalar." << std::endl;
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

      if( !field_dim.has_value() )
      {
        ldbg << "Please provide a field dimension using field_dim slot " << std::endl;
      }
      const unsigned long ncomps = *field_dim;

      if( !grid_subdiv.has_value() )
      {
        ldbg << "Please provide a subdivision value using grid_subdiv slot " << std::endl;
      }
      const long cell_subdiv = *grid_subdiv;
      
      const IJK subcell_grid = { cell_subdiv, cell_subdiv, cell_subdiv };
      const ssize_t stride = cell_subdiv*cell_subdiv*cell_subdiv;
      
      if( ! grid_cell_values->has_field(*field_name) )
        {
          ldbg << "Create cell field "<< *field_name << " subdiv="<<cell_subdiv<<" ncomps="<<ncomps<< std::endl
               << std::endl;
          grid_cell_values->add_field(*field_name,cell_subdiv,ncomps);
        }
      assert( size_t(cell_subdiv) == grid_cell_values->field(*field_name).m_subdiv );
      assert( stride * ncomps == grid_cell_values->field(*field_name).m_components );
      auto field_data = grid_cell_values->field_data(*field_name);
      
      if( file_name->empty() )
      {
        fatal_error() << "Please provide a file name." << std::endl;
      }
      std::ifstream file(*file_name);
      if (!file.is_open()) {
        throw std::runtime_error(std::string("Could not open file ") + *file_name);
      }

      int Nx, Ny, Nz, dataDimension;
      std::string dataType;
      
      [[maybe_unused]] const IJK dims = grid_cell_values->grid_dims();
      const IJK dom_dims = domain->grid_dimension();
      
      std::string line;
      bool lookup_table_found = false;
      ssize_t pointCount;
      
      while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;
        if (keyword == "DIMENSIONS") {
          lineStream >> Nx >> Ny >> Nz;
          
          if ( (dom_dims.i != (Nx/cell_subdiv) ) || (dom_dims.j != (Ny/cell_subdiv) ) || (dom_dims.k != (Nz/cell_subdiv) ) ) {
            fatal_error() << "VTK file grid dimensions (without subdiv) are not equal to simulation domain grid dimensions" << std::endl
                          << "Simulation domain  grid dimensions = " << dom_dims << std::endl
                          << "VTK file grid dimensions = " << IJK{Nx/cell_subdiv,Ny/cell_subdiv,Nz/cell_subdiv} << std::endl;
          }
        } else if (keyword == "POINT_DATA") {
          lineStream >> pointCount;
          assert(pointCount == Nx*Ny*Nz);
          assert((dims.i*dims.j*dims.k*stride) == pointCount);
        } else if (keyword == "SCALARS" || keyword == "VECTORS") {
          dataType = keyword;
          std::string name, type;
          lineStream >> name >> type;
          dataDimension = (keyword == "SCALARS") ? 1 : 3; // Scalars have 1 dimension, Vectors have 3
          if (dataDimension != *field_dim)
            throw std::runtime_error(std::string("Dimension != 1. Field should be scalar."));
        } else if (line.find("LOOKUP_TABLE") != std::string::npos) {
          lookup_table_found = true;
          ldbg << "LOOKUP_TABLE found" << std::endl;
          break;
        }
      }
    
      if (!lookup_table_found) {
        fatal_error() << "LOOKUP_TABLE not found in the file." << std::endl;
      }
      
      ldbg << "Using " << to_cstr( *grid_ordering ) << " cell ordering"<< std::endl;
      
      const IJK gcv_grid = grid_cell_values->grid_dims();
      
      // Read the lookup table values and reorder on the fly using a single loop
      for (ssize_t idx = 0; idx < pointCount; ++idx) {
        double value;
        if (!(file >> value)) {
          fatal_error() << "Failed to read data from the file." << std::endl;
        }

        // Position in the global VTK grid
        IJK vtk_location;
        if ( *grid_ordering == StructuredGridOrdering::C_ORDER )
          {
            // In row-major ('C' style), flatten_array is generated as :
            // for i in range(Nx):
            //    for j in range(Ny):
            //        for k in range(Nz):
            //            N = i * Ny * Nz + j * Nz + k
            // i,j,k index can be recovered as :
            // i = idx // (Ny * Nz)
            // j = (idx % (Ny * Nz)) // Nz
            // k = idx % Nz
            ssize_t i = idx / ( Ny * Nz );
            ssize_t j = ( idx % ( Ny * Nz ) / Nz );
            ssize_t k = idx % Nz;
            vtk_location = IJK { i, j, k };
          }
        else if ( *grid_ordering == StructuredGridOrdering::F_ORDER )
          {
            // In column-major ('F' style), flatten_array is generated as :
            // for k in range(Nz):
            //    for j in range(Ny):
            //        forik in range(Nx):
            //            N = k * Ny * Nx + j * Nx + i
            // i,j,k index can be recovered as :
            // k = N // (Ny * Nx)
            // j = (N % (Ny * Nx)) // Nx
            // i = N % Nx
            ssize_t k = idx / ( Ny * Nx );
            ssize_t j = ( idx % ( Ny * Nx ) / Nx );
            ssize_t i = idx % Nx;
            vtk_location = IJK { i, j, k };            
          }
        //        IJK vtk_location = IJK { idx / (Nz * Ny), (idx / Nz) % Ny, idx % Nz };
        
        // Position in the global domain grid
        IJK dom_location = vtk_location / subcell_grid;
        // Position in the MPI domain grid
        IJK mpi_location = dom_location - grid->offset();
        // Position in the subcell grid
        IJK subcell_location = vtk_location % subcell_grid;

        // Computing main cell index
        ssize_t cell_index = grid_ijk_to_index( gcv_grid, mpi_location);
        // Computing sub cell index
        ssize_t subcell_index = grid_ijk_to_index( subcell_grid , subcell_location );
        // Computing destination index
        ssize_t dest_idx = cell_index * stride + subcell_index;
        
        // Check if the mpi_location is in my rank domain and assign value to data ptr
        if (grid->contains(mpi_location)) field_data.m_data_ptr[dest_idx] = value;
        
      }
      file.close();
    }

    inline std::string documentation() const override final
    {
      return R"EOF(
This operator reads a scalar field from an external .vtk file (structured grid) and assigns the scalar value to the points on a Cartesian grid aligned with the simulation domain. The grid dimensions on which the scalar field is defined needs to be either the same of the domain grid or a multiple (using grid_subdiv) of it. In addition, using grid_ordering (default = "F_ORDER"), one can specify how the input .vtk file was written, e.g. using "F_ORDER" or "C_ORDER". This operators outputs a grid_cell_values data structure that can be subsequently used as a mask for example to populate the domain with particles.

Usage example:

  - read_cell_values:
      field_name: "MASK1"
      file_name: "points_40x40x40.vtk"
      grid_subdiv: 4
      grid_ordering: F_ORDER
  - lattice:
      structure: BCC
      types: [ W, W ]
      size: [ 3 ang , 3 ang , 3 ang ]
      grid_cell_mask_name: MASK1
      grid_cell_mask_value: 1

Header of points_40x40x40.vtk file:

# vtk DataFile Version 5.1
vtk output
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 40 40 40
SPACING 1 1 1
ORIGIN 0 0 0
POINT_DATA 64000
SCALARS ScalarData vtktypeint64
LOOKUP_TABLE default
0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 

)EOF";
    }
    
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(read_cell_values)
  {
    OperatorNodeFactory::instance()->register_factory( "read_cell_values", make_grid_variant_operator<ReadCellValues> );
  }

}

