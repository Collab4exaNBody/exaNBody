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

#include <iostream>
#include <fstream>

#include <exanb/core/log.h>

#include <exanb/io/vtk_writer.h>

#include <exanb/grid_cell_particles/grid_particle_field_accessor.h>

namespace exanb
{
  

  template<typename GridT, typename FType, class GridAccT = exanb::GridParticleFieldAccessor<const typename GridT::CellParticles *> >
  inline void write_ascii_datas_from_field(const GridT& grid, FType ftype, const std::string& name, const std::string& type, std::ofstream& file_vtp, bool is_ghosts, GridAccT gridacc = {nullptr} )
  {
    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    // cannot initalize default grid accessor right in the parameter list (compiler says 'grid may not appear in this context)
    // so we detect default grid accessor and initializes it with correct cells value from grid
    if constexpr ( std::is_same_v<GridAccT,exanb::GridParticleFieldAccessor<const typename GridT::CellParticles *> > )
    {
      if( gridacc.m_cells == nullptr ) gridacc.m_cells = cells;
    }

    using field_type = typename FType::value_type;
    //using comp_type = typename ParaViewTypeId<field_type>::comp_type;

    if ( ParaViewTypeId<field_type>::ncomp == 0 ) return;

    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\"";
    if( ParaViewTypeId<field_type>::ncomp > 1 ) file_vtp << " NumberOfComponents=\""<< ParaViewTypeId<field_type>::ncomp <<"\"";
    file_vtp << " format=\"ascii\">"<< std::endl;
    file_vtp << vtk_space_offset_ten;
    
    for(size_t c=0; c<n_cells;++c)
    {
      if( !grid.is_ghost_cell(c) || is_ghosts )
      {
        //[[maybe_unused]] const auto field_ptr = cells[c][ftype];
        size_t np = cells[c].size();
        for(size_t pos=0;pos<np;++pos)
        {
          [[maybe_unused]] const auto value = gridacc.get(c,pos,ftype);
          if constexpr ( ParaViewTypeId<field_type>::ncomp == 1 )
          {
            if ( std::is_integral_v<field_type> ) file_vtp << ' ' << (int64_t) value ;
            else file_vtp << ' ' << value;
          }
          if constexpr ( std::is_same_v<field_type,Vec3d> )
          {
            file_vtp << ' ' << value.x ;
            file_vtp << ' ' << value.y ;
            file_vtp << ' ' << value.z ;
          }
          if constexpr ( std::is_same_v<field_type,Mat3d> )
          {
            file_vtp << ' ' << value.m11 ;
            file_vtp << ' ' << value.m12 ;
            file_vtp << ' ' << value.m13 ;
            file_vtp << ' ' << value.m21 ;
            file_vtp << ' ' << value.m22 ;
            file_vtp << ' ' << value.m23 ;
            file_vtp << ' ' << value.m31 ;
            file_vtp << ' ' << value.m32 ;
            file_vtp << ' ' << value.m33 ;
          }
        }
      }
    }
    file_vtp << std::endl <<  vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  template<typename GridT, typename FType_x, typename FType_y, typename FType_z>
  inline void write_ascii_datas_from_fields(const GridT& grid, const FType_x& ftype_x, const FType_y& ftype_y, const FType_z& ftype_z, const std::string& name, const std::string& type, std::ofstream& file_vtp, bool is_ghosts)
  {
    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\" NumberOfComponents=\"3\"  format=\"ascii\">"<< std::endl;

    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    file_vtp << vtk_space_offset_ten;
    for(size_t c=0; c<n_cells;++c)
    {
      if( !grid.is_ghost_cell(c) || is_ghosts )
      {
        const auto * __restrict__ field_x_ptr = cells[c].field_pointer_or_null(ftype_x);
        const auto * __restrict__ field_y_ptr = cells[c].field_pointer_or_null(ftype_y);
        const auto * __restrict__ field_z_ptr = cells[c].field_pointer_or_null(ftype_z);
        size_t np = cells[c].size();
        for(size_t pos=0;pos<np;++pos)
        {
          file_vtp << std::to_string(field_x_ptr[pos]) << " ";
          file_vtp << std::to_string(field_y_ptr[pos]) << " ";
          file_vtp << std::to_string(field_z_ptr[pos]) << " ";
        }
      }
    }
    file_vtp << std::endl
             <<  vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  inline void write_ascii_datas_from_int(const int64_t data, const std::string& name, std::ofstream& file_vtp)
  {
    file_vtp <<  vtk_space_offset_eight << "<DataArray type=\"Int64\" Name=\"" << name << "\"  format=\"ascii\">"<< std::endl;

    file_vtp << vtk_space_offset_ten;
    file_vtp << std::to_string(data) << std::endl;
    file_vtp <<  vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  // case of positions : we need to do an exception because xform (deformation of the box)
  template<typename GridT>
  inline void write_ascii_positions(const GridT& grid, const std::string& name, const std::string& type, std::ofstream& file_vtp, bool is_ghosts, const Mat3d& xform)
  {
    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\" NumberOfComponents=\"3\"  format=\"ascii\">"<< std::endl;

    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    file_vtp << vtk_space_offset_ten;
    for(size_t c=0; c<n_cells;++c)
    {
      if( !grid.is_ghost_cell(c) || is_ghosts )
      {
        const auto * __restrict__ rx = cells[c].field_pointer_or_null(field::rx);
        const auto * __restrict__ ry = cells[c].field_pointer_or_null(field::ry);
        const auto * __restrict__ rz = cells[c].field_pointer_or_null(field::rz);
        size_t np = cells[c].size();
        for(size_t pos=0;pos<np;++pos)
        {
          Vec3d pos_vec = {rx[pos],ry[pos],rz[pos]};

          pos_vec = xform * pos_vec;

          file_vtp << std::to_string(pos_vec.x) << " ";
          file_vtp << std::to_string(pos_vec.y) << " ";
          file_vtp << std::to_string(pos_vec.z) << " ";
        }
      }
    }
    file_vtp << std::endl
             <<  vtk_space_offset_eight << "</DataArray>" << std::endl;
  }
}
