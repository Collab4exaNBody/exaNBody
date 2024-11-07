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
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/domain.h>
#include <onika/physics/units.h>

#include <exanb/io/vtk_writer.h>
#include <exanb/io/vtk_writer_binary.h>
#include <exanb/io/vtk_writer_ascii.h>

#include <iostream>
#include <fstream>
#include <string>

namespace exanb
{
  

  namespace vtklegacy_details
  {
    // Convert bigEndian littleEndian
    // Thanks to https://stackoverflow.com/questions/105252
    template <typename T>
    void SwapEnd(T& var)
    {
      char* varArray = reinterpret_cast<char*>(&var);
      for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
        std::swap(varArray[sizeof(var) - 1 - i],varArray[i]);
    }

    struct WriteArrayData
    {
      std::ofstream& out;
      bool binary = true;
      bool ghost = false;
      template<class GridT, class FidT>
      inline void operator () ( GridT& grid, const FidT& fid )
      {
        using field_type = typename FidT::value_type;      
        
        if( ParaViewTypeId<field_type>::ncomp == 3 )
        {
          out << "VECTORS "<< fid.short_name() <<" double" << std::endl;
        }
        else if( ParaViewTypeId<field_type>::ncomp == 1 )
        {
          out<< "SCALARS "<< fid.short_name() <<" "<< ( std::is_integral_v<field_type> ? "int" : "double" ) <<" 1" << std::endl<< "LOOKUP_TABLE default" << std::endl;
        }
        else return;

        size_t n_cells = grid.number_of_cells();
        auto cells = grid.cells();
        for(size_t c=0; c<n_cells;++c)
        {
          if( !grid.is_ghost_cell(c) || ghost)
          {
            [[maybe_unused]] const auto * __restrict__ field_ptr = cells[c].field_pointer_or_null( fid );
            size_t np = cells[c].size();
            assert( np==0 || field_ptr!=nullptr );
            for(size_t pos=0;pos<np;++pos)
            {
              if constexpr ( ParaViewTypeId<field_type>::ncomp == 1 )
              {
                if( std::is_integral_v<field_type> )
                {
                  int i = field_ptr[pos];
                  if( binary ) { SwapEnd(i); out.write(reinterpret_cast<char*>(&i), sizeof(int)); }
                  else { out << i << std::endl; }
                }
                else
                {
                  double d = field_ptr[pos];
                  if( binary ) { SwapEnd(d); out.write(reinterpret_cast<char*>(&d), sizeof(double)); }
                  else { out << d << std::endl; }
                }
              }
              if constexpr ( std::is_same_v<field_type,Vec3d> )
              {
                double rx = field_ptr[pos].x;
                double ry = field_ptr[pos].y;
                double rz = field_ptr[pos].z;
                if( binary )
                {
                  SwapEnd(rx); SwapEnd(ry); SwapEnd(rz);
                  out.write(reinterpret_cast<char*>(&rx), sizeof(double));
                  out.write(reinterpret_cast<char*>(&ry), sizeof(double));
                  out.write(reinterpret_cast<char*>(&rz), sizeof(double));
                }
                else
                {
                  out << rx<<' '<<ry<<' '<<rz<<std::endl;
                }
              }

            }
          }
        }
        out << std::endl;
      }
    };
    
  }

  template<typename GridT>
  class VtkLegacyWriter : public OperatorNode
  {
    ADD_SLOT( GridT       , grid       , INPUT );
    ADD_SLOT( std::string , filename   , INPUT );
    ADD_SLOT( bool        , ghost      , INPUT , false);
    ADD_SLOT( bool        , ascii      , INPUT , false);

  public:
    inline void execute () override final
    {
      using namespace vtklegacy_details;
    
      size_t n_cells = grid->number_of_cells();
      auto cells = grid->cells();
      uint64_t nb_particles = 0;
      for(size_t c=0;c<n_cells;++c)
      {
        if( ! grid->is_ghost_cell(c) || *ghost )
        {
          nb_particles += grid->cell_number_of_particles(c);
        }
      }
    
      ldbg << "write file "<< *filename << ", ascii="<<*ascii <<std::endl;

      std::ofstream file;
      file.open(*filename);

      file << "# vtk DataFile Version 2.0" << std::endl
           << "ExaNB VTK legacy exporter" << std::endl
           << ( (*ascii)? "ASCII" : "BINARY" ) << std::endl
           << "DATASET POLYDATA" << std::endl
           << "POINTS " << nb_particles
           << " double"  << std::endl;
      
      file << std::scientific << std::setprecision(10);
      
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        if( ! grid->is_ghost_cell(cell_i) || *ghost )
        {
          size_t n = cells[cell_i].size();
          for(size_t i =0; i<n;++i)
          {
            double rx = cells[cell_i][field::rx][i];
            double ry = cells[cell_i][field::ry][i];
            double rz = cells[cell_i][field::rz][i];
            if( *ascii )
            {
              file << rx<<' '<<ry<<' '<<rz<<std::endl;
            }
            else
            {
              SwapEnd(rx); SwapEnd(ry); SwapEnd(rz);
              file.write(reinterpret_cast<char*>(&rx), sizeof(double));
              file.write(reinterpret_cast<char*>(&ry), sizeof(double));
              file.write(reinterpret_cast<char*>(&rz), sizeof(double));
            }
          }
        }
      }


      file << std::endl << "VERTICES " << nb_particles << " "<< nb_particles*2 << std::endl;
      for(size_t i=0; i<nb_particles;++i)
      {
        int32_t num = 1;
        int32_t idx = i;
        if( *ascii )
        {
          file << num<<' '<<idx<<std::endl;
        }
        else
        {
          SwapEnd(num);
          SwapEnd(idx);
          file.write(reinterpret_cast<char*>(&num), sizeof(int32_t));        
          file.write(reinterpret_cast<char*>(&idx), sizeof(int32_t));
        }
      }

      file << std::endl;      
      file << "POINT_DATA " << nb_particles << std::endl;

      apply_grid_field_set( *grid, WriteArrayData{file,!(*ascii),*ghost}, GridT::field_set );

      file.close();
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "write_vtklegacy",make_grid_variant_operator<VtkLegacyWriter>);
  }

}
