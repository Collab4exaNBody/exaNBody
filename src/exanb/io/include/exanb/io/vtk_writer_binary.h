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

#include <zlib.h>//lib for compression

#include <exanb/core/log.h>
#include "base-n/basen.h"//Base64 compression

#include <exanb/io/vtk_writer.h>

#include <exanb/core/grid_particle_field_accessor.h>

namespace exanb
{
  

  template<typename ArrayT>
  inline void write_binary_datas(std::ofstream& file_vtp, const int compression_level, ArrayT& sources)
  {

    //debug<FType> d;
    //debug<soatl::FieldId<exanb::field::_type>::value_type> d1;


    // Compression with zlib
    // https://stackoverflow.com/questions/4538586/how-to-compress-a-buffer-with-zlib
    unsigned char *sources_datas = reinterpret_cast<unsigned char*>(sources.data());
    size_t sources_length = sources.size()*sizeof(decltype(sources[0]));

    //#warning FIXME: compressBound not found ???
    unsigned long destination_length_ul = compressBound(sources_length);
    
    //unsigned char* destination_datas = new unsigned char[static_cast<size_t>(destination_length_ul)];
    std::unique_ptr<unsigned char[]> destination_datas(new unsigned char[destination_length_ul]);

    // For safety
    for(size_t d=0;d<destination_length_ul;++d)
    {
      destination_datas[d] = 0;
    }

    int is_succes = compress2(destination_datas.get(),
                              &destination_length_ul,
                              sources_datas, static_cast<unsigned long>(sources_length), compression_level);

    if(is_succes != Z_OK)
      {
        lerr << "Impossible to compress datas for vtp files." << std::endl;
        std::abort();
      }
    //----------------------------------------------------------------------------------------


    // Header informations
    // 1 : number of blocks
    // 2 : blocks size
    // 3 : last block size (if different)
    // 4 : compress datas size
    std::vector<uint64_t> header_datas(4,0);
    header_datas[0] = static_cast<uint64_t>(1);
    header_datas[1] = static_cast<uint64_t>(sources_length);
    header_datas[2] = static_cast<uint64_t>(0);
    header_datas[3] = static_cast<uint64_t>(destination_length_ul);
    //----------------------------------------------------------------------------------------

    // Encoding in base64 header and datas
    std::string buffer;
    buffer  = vtk_space_offset_ten;
    bn::encode_b64(reinterpret_cast<char*>(header_datas.data()),
                   reinterpret_cast<char*>(header_datas.data())+4*sizeof(uint64_t),
                   std::back_inserter(buffer));
    buffer += "=";// use as separation by paraview
    bn::encode_b64(reinterpret_cast<char*>(destination_datas.get()),
                   reinterpret_cast<char*>(destination_datas.get())+static_cast<size_t>(destination_length_ul),
                   std::back_inserter(buffer));
    buffer += "==";// use as separation by paraview. WHY == HERE ??????
    //----------------------------------------------------------------------------------------

    file_vtp << buffer;
  }

  template<class GridT, class CellsAccessorT, class FType >
  inline void write_binary_datas_from_field(const GridT& grid, CellsAccessorT cells, FType ftype, const std::string& name, const std::string& type, std::ofstream& file_vtp, const int compression_level, bool is_ghosts )
  {
    size_t n_cells = grid.number_of_cells();

    using field_type = typename FType::value_type;
    using comp_type = typename ParaViewTypeId<field_type>::comp_type;

    if ( ParaViewTypeId<field_type>::ncomp == 0 ) return;

    std::vector<comp_type> sources;
    for(size_t c=0; c<n_cells;++c)
    {
      if( !grid.is_ghost_cell(c) || is_ghosts)
      {
        // [[maybe_unused]] const auto field_ptr = cells[c][ftype];
        // assert( field_ptr != nullptr );
        size_t np = cells[c].size();
        for(size_t pos=0;pos<np;++pos)
        {
          [[maybe_unused]] const auto value = cells[c][ftype][pos];
          if constexpr ( ParaViewTypeId<field_type>::ncomp == 1 )
          {
            sources.push_back( value );
          }
          if constexpr ( std::is_same_v<field_type,Vec3d> )
          {
            sources.push_back( value.x );
            sources.push_back( value.y );
            sources.push_back( value.z );
          }
          if constexpr ( std::is_same_v<field_type,Quaternion> )
          {
            sources.push_back( value.w );
            sources.push_back( value.x );
            sources.push_back( value.y );
            sources.push_back( value.z );
          }
          if constexpr ( std::is_same_v<field_type,Mat3d> )
          {
            sources.push_back( value.m11 );
            sources.push_back( value.m12 );
            sources.push_back( value.m13 );
            sources.push_back( value.m21 );
            sources.push_back( value.m22 );
            sources.push_back( value.m23 );
            sources.push_back( value.m31 );
            sources.push_back( value.m32 );
            sources.push_back( value.m33 );
          }
        }
      }
    }

    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\"";
    if( ParaViewTypeId<field_type>::ncomp > 1 ) file_vtp << " NumberOfComponents=\""<< ParaViewTypeId<field_type>::ncomp <<"\"";
    file_vtp << " format=\"binary\">"<< std::endl;
    write_binary_datas(file_vtp, compression_level, sources);
    file_vtp << std::endl << vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  inline void write_binary_datas_from_int(const int data, const std::string& name, std::ofstream& file_vtp, const int compression_level)
  {
    std::vector<int> sources(data, 1);

    file_vtp << vtk_space_offset_eight << "<DataArray type=\"Int32\" Name=\"" << name << "\" format=\"binary\">"<< std::endl;
    write_binary_datas(file_vtp, compression_level, sources);
    file_vtp << std::endl
             << vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  template<typename GridT, typename FType_x, typename FType_y, typename FType_z>
  inline void write_binary_datas_from_fields(const GridT& grid, const FType_x& ftype_x, const FType_y& ftype_y, const FType_z& ftype_z, const std::string& name, const std::string& type, std::ofstream& file_vtp, const int compression_level, bool is_ghosts)
  {
    //compilation time test
    static_assert(std::is_same<typename FType_x::value_type,typename FType_y::value_type>::value, "write_binary_datas_from_fields bad types");
    static_assert(std::is_same<typename FType_x::value_type,typename FType_z::value_type>::value, "write_binary_datas_from_fields bad types");

    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    std::vector<typename FType_x::value_type> sources;

    for(size_t c=0; c<n_cells;++c)
    {
      //Check if cell is a ghost cell
      if(!grid.is_ghost_cell(c) || is_ghosts)
      {
        const auto * __restrict__ field_x_ptr = cells[c].field_pointer_or_null(ftype_x);
        const auto * __restrict__ field_y_ptr = cells[c].field_pointer_or_null(ftype_y);
        const auto * __restrict__ field_z_ptr = cells[c].field_pointer_or_null(ftype_z);
        size_t np = cells[c].size();
        for(size_t pos=0;pos<np;++pos)
        {
          sources.push_back(field_x_ptr[pos]);
          sources.push_back(field_y_ptr[pos]);
          sources.push_back(field_z_ptr[pos]);
        }
      }
    }

    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\"  NumberOfComponents=\"3\" format=\"binary\">"<< std::endl;
    write_binary_datas(file_vtp, compression_level, sources);
    file_vtp << std::endl
             << vtk_space_offset_eight << "</DataArray>" << std::endl;
  }

  // case of positions : we need to do an exception because xform (deformation of the box)
  template<typename GridT>
  inline void write_binary_positions(const GridT& grid, const std::string& name, const std::string& type, std::ofstream& file_vtp, const int compression_level, bool is_ghosts, const Mat3d& xform)
  {
    size_t n_cells = grid.number_of_cells();
    auto cells = grid.cells();

    std::vector<double> sources;

    for(size_t c=0; c<n_cells;++c)
    {
      //Check if cell is a ghost cell
      if(!grid.is_ghost_cell(c) || is_ghosts)
      {
        size_t np = cells[c].size();
        const auto * __restrict__ rx = cells[c].field_pointer_or_null(field::rx);
        const auto * __restrict__ ry = cells[c].field_pointer_or_null(field::ry);
        const auto * __restrict__ rz = cells[c].field_pointer_or_null(field::rz);
        for(size_t pos=0;pos<np;++pos)
        {
          Vec3d pos_vec = {rx[pos],ry[pos],rz[pos]};
          pos_vec = xform * pos_vec;
          sources.push_back(pos_vec.x);
          sources.push_back(pos_vec.y);
          sources.push_back(pos_vec.z);
        }
      }
    }

    file_vtp << vtk_space_offset_eight << "<DataArray type=\""<< type << "\" Name=\"" << name << "\"  NumberOfComponents=\"3\" format=\"binary\">"<< std::endl;
    write_binary_datas(file_vtp, compression_level, sources);
    file_vtp << std::endl
             << vtk_space_offset_eight << "</DataArray>" << std::endl;
  }
}
