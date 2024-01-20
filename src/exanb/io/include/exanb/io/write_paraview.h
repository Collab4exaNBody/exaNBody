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
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/quantity.h>
#include <exanb/core/string_utils.h>

#include <exanb/compute/field_combiners.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <exanb/io/vtk_writer.h>
#include <exanb/io/vtk_writer_binary.h>
#include <exanb/io/vtk_writer_ascii.h>

#include <iostream>
#include <fstream>
#include <mpi.h>
#include <string>
#include <iomanip>
#include <filesystem>

namespace exanb
{
  namespace ParaviewWriteTools
  {
  
    struct WriteScalarList
    {
      std::ofstream& out;
      std::function<bool(const std::string&)> m_field_selector;
      bool first = true;
      template<class GridT, class FidT>
      inline void operator () ( GridT& , const FidT& fid )
      {
        using field_type = typename FidT::value_type;      
        if( ParaViewTypeId<field_type>::ncomp==1 && m_field_selector(fid.short_name()) )
        {
          if( first ) first=false; else out<<", ";
          out << fid.short_name() ;        
        }
      }
    };

    struct WriteVectorList
    {
      std::ofstream& out;
      std::function<bool(const std::string&)> m_field_selector;
      bool first = true;
      template<class GridT, class FidT>
      inline void operator () ( GridT& , const FidT& fid )
      {
        using field_type = typename FidT::value_type;      
        if( ParaViewTypeId<field_type>::ncomp>1 && m_field_selector(fid.short_name()) )
        {
          if( first ) first=false; else out<<", ";
          out << fid.short_name() ;        
        }
      }
    };
    
    struct WriteArrayDecl
    {
      std::ofstream& out;
      std::function<bool(const std::string&)> m_field_selector;
      template<class GridT, class FidT>
      inline void operator () ( GridT& , const FidT& fid )
      {
        using field_type = typename FidT::value_type; 
        if( m_field_selector(fid.short_name()) )
        {
          if( ParaViewTypeId< field_type >::ncomp == 1 )
          {
            out << vtk_space_offset_six << "<PDataArray type=\""<< ParaViewTypeId<field_type>::str() <<"\" Name=\""<<fid.short_name()<<"\"/>" << std::endl;
          }
          if( ParaViewTypeId< field_type >::ncomp > 1 )
          {
            out << vtk_space_offset_six << "<PDataArray type=\""<< ParaViewTypeId<field_type>::str() <<"\" Name=\""<<fid.short_name()<<"\" NumberOfComponents=\""<< ParaViewTypeId<field_type>::ncomp << "\"/>" << std::endl;
          }
        }
      }
    };

    template<class GridAccessorT>
    struct WriteArrayData
    {
      const GridAccessorT& m_gridacc;
      std::ofstream& out;
      std::function<bool(const std::string&)> m_field_selector;
      int compression_level=1;
      bool binary = true;
      bool ghost = false;
      template<class GridT, class FidT>
      inline void operator () ( GridT& grid, const FidT& fid )
      {
        using field_type = typename FidT::value_type;      
        if( m_field_selector(fid.short_name()) )
        {
          if(binary)
          {
            write_binary_datas_from_field(grid, fid , fid.short_name() , ParaViewTypeId<field_type>::str() , out, compression_level, ghost , m_gridacc );
          }
          else
          {
            write_ascii_datas_from_field(grid, fid , fid.short_name() , ParaViewTypeId<field_type>::str() , out, ghost , m_gridacc );
          }
        }
      }
    };
    template<class GridAccessorT> inline WriteArrayData<GridAccessorT> make_paraview_array_data_writer( const GridAccessorT& gridacc, std::ofstream& out, std::function<bool(const std::string&)> fs, int compression_level, bool binary, bool ghost)
    {
      return { gridacc, out , fs, compression_level, binary, ghost };
    }

    template<class LDBG, class GridT, class GridAccesorT, class FieldSelectorT, class... GridFields >
    static inline void write_particles(LDBG& ldbg,
              MPI_Comm comm, const GridT& grid, const GridAccesorT& gridacc, const Domain& domain, const std::string& filename, const FieldSelectorT& field_selector,
              const std::string& compression, bool binary_mode, bool write_box, bool write_external_box, bool write_ghost, const GridFields& ... grid_fields )
    {    
      Mat3d xform = domain.xform();

      // Gestion of compression level for binary datas
      int compression_level = -1;
      if(compression == "default")  compression_level = -1;
      else if(compression == "min") compression_level = 0;
      else if(compression == "max") compression_level = 9;
      else if(compression>="0" && compression<="9") compression_level = std::stoi(compression);
      else
      {
        fatal_error() << "ParaviewWriter : compression type is unknow. Keywords : max, min, default or [0-9]" << std::endl;
      }
      
      ldbg << "using compression level "<<compression_level<<std::endl;

      // MPI Initialization
      int rank=0, np=1;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &np);
      
      MPI_Barrier( comm );

      // only one proc need to write .ptvp file
      if(rank==0)
      {
        std::filesystem::create_directories( filename );
        std::string filename_pvtp = (filename) + ".pvtp";
        ldbg << "write " << filename_pvtp << std::endl;

        std::ofstream file_pvtp(filename_pvtp);
        if( ! file_pvtp.good() )
        {
          fatal_error()<<"Can't open file "<<filename_pvtp<<" for writing"<<std::endl;
        }
        file_pvtp << "<?xml version=\"1.0\"?>" << std::endl
                  << vtk_space_offset_two << "<VTKFile type=\"PPolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">" << std::endl
                  << vtk_space_offset_four << "<PPolyData GhostLevel=\"0\">" << std::endl
                  << vtk_space_offset_four << "<PPointData Scalar=\"" ;
        apply_grid_fields( grid, WriteScalarList{file_pvtp,field_selector}, grid_fields... );
        if(write_ghost) file_pvtp <<", ghost";
        file_pvtp << "\" Vectors=\"" ;
        apply_grid_fields( grid, WriteVectorList{file_pvtp,field_selector}, grid_fields... ); 
        file_pvtp << "\">" << std::endl;

        apply_grid_fields( grid, WriteArrayDecl{file_pvtp,field_selector}, grid_fields... );            
          
        if(write_ghost)
        {
          file_pvtp << vtk_space_offset_six << "<PDataArray type=\"UInt8\" Name=\"ghost\"/>" << std::endl;
        }

        file_pvtp << vtk_space_offset_four << "</PPointData>" << std::endl
                  << vtk_space_offset_four << "<PPoints>" << std::endl
                  << vtk_space_offset_six << "<PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl
                  << vtk_space_offset_four << "</PPoints>" << std::endl;

        for(int i=0;i<np; ++i)
        {
          const auto last_slash = filename.rfind('/');
          std::string subdir = filename;
          if( last_slash != std::string::npos )
          {
            subdir = filename.substr(last_slash+1) + "/";
          }
          const std::string filename_vtp = format_string("%s/%09d.vtp",subdir,i);
          file_pvtp << vtk_space_offset_four << "<Piece Source=\""<< filename_vtp <<"\"/>" << std::endl;
        }

        file_pvtp << vtk_space_offset_two << "</PPolyData>" << std::endl << "</VTKFile>" << std::endl;
        file_pvtp.close();
      }
      
      // every one waits until output directory is created
      MPI_Barrier(comm);


      //-------------------------------------------------------------------------------------------
      // Starting to write .vtp file by each proc -------------------------------------------------
      size_t n_cells = grid.number_of_cells();
      auto cells = grid.cells();
      size_t nb_particles = grid.number_of_particles();
      if(!write_ghost) { nb_particles -= grid.number_of_ghost_particles(); }

      const std::string filename_vtp = format_string("%s/%09d.vtp",filename,rank);
      ldbg << "write " << filename_vtp << std::endl;

      std::ofstream file_vtp(filename_vtp);
      if( ! file_vtp.good() )
      {
        fatal_error()<<"Can't open file "<<filename_vtp<<" for writing"<<std::endl;
      }
      file_vtp << "<?xml version=\"1.0\"?>" << std::endl
               << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">" << std::endl
               <<  vtk_space_offset_two  << "<PolyData>" << std::endl
               <<  vtk_space_offset_four << "<Piece  NumberOfPoints=\""<< nb_particles << "\" NumberOfVerts=\"1\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">" << std::endl
               <<  vtk_space_offset_six  << "<PointData>" << std::endl;

      apply_grid_fields( grid, make_paraview_array_data_writer(gridacc,file_vtp,field_selector,compression_level,binary_mode,write_ghost) , grid_fields... );

      if(write_ghost)
      {
        ldbg << "write ghost" << std::endl;
        if(binary_mode)
        {
          std::vector<uint8_t> sources;
          for(size_t c=0; c<n_cells;++c)
          {
            if(grid.is_ghost_cell(c)) for(size_t pos=0;pos<cells[c].size();++pos) sources.push_back(1);
            else for(size_t pos=0;pos<cells[c].size();++pos) sources.push_back(0);
          }
          file_vtp <<  vtk_space_offset_eight << "<DataArray type=\"UInt8\" Name=\"ghost\" format=\"binary\">"<< std::endl;
          write_binary_datas(file_vtp, compression_level, sources);
        }
        else
        {
          file_vtp <<  vtk_space_offset_eight << "<DataArray type=\"UInt8\" Name=\"ghost\" format=\"ascii\">" << std::endl;
          for(size_t c=0; c<n_cells;++c)
          {
            if(grid.is_ghost_cell(c)) for(size_t pos=0;pos<cells[c].size();++pos) file_vtp << ' ' << 1;
            else for(size_t pos=0;pos<cells[c].size();++pos) file_vtp << ' ' << 0;
          }
        }
        file_vtp << std::endl <<  vtk_space_offset_eight << "</DataArray>" << std::endl;
      }
      
      file_vtp <<  vtk_space_offset_six << "</PointData>"  << std::endl
               <<  vtk_space_offset_six << "<CellData>" << std::endl
               <<  vtk_space_offset_six << "</CellData>" << std::endl
               <<  vtk_space_offset_six << "<Points>" << std::endl;

      ldbg << "write positions" << std::endl;
      if(binary_mode)
      {
        write_binary_positions(grid, std::string(""), std::string("Float64"), file_vtp, compression_level, write_ghost, xform);
      }
      else
      {
        write_ascii_positions(grid, std::string(""), std::string("Float64"), file_vtp, write_ghost, xform);
      }
      file_vtp <<  vtk_space_offset_six << "</Points>" << std::endl
               <<  vtk_space_offset_six << "<Verts>" << std::endl;
               
      ldbg << "write connectivity" << std::endl;
      if(binary_mode)
      {
        std::vector<int64_t> sources;
        for(size_t c=0; c<n_cells;++c)
        {
          if( !grid.is_ghost_cell(c) || write_ghost )
          {
            for(size_t pos=0;pos<cells[c].size();++pos) sources.push_back( sources.size() );
          }
        }

        file_vtp <<  vtk_space_offset_eight << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\">"<< std::endl;
        write_binary_datas(file_vtp, compression_level, sources);
      }
      else
      {
        file_vtp <<  vtk_space_offset_eight << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
        file_vtp << vtk_space_offset_ten;
        for(size_t i=0; i<nb_particles;++i)
        {
          file_vtp << ' ' << i ;
        }
      }
      file_vtp << std::endl <<  vtk_space_offset_eight << "</DataArray>" << std::endl;

      ldbg << "write offsets" << std::endl;
      write_ascii_datas_from_int(nb_particles, std::string("offsets"), file_vtp);

      file_vtp <<  vtk_space_offset_six  << "</Verts>"<< std::endl
               <<  vtk_space_offset_four << "</Piece>"<< std::endl
               <<  vtk_space_offset_two  << "</PolyData>"<< std::endl
               << "</VTKFile>"<< std::endl;
      file_vtp.close();
      //Stop writing .vtp file---------------------------------------------------------------------


      //----------------------------------------------BOX------------------------------------------
      // Simulation box
      // TO DO : box will be dependant of the time
      if(write_box && rank==0)
        {
          ldbg << "write box" << std::endl;
          std::stringstream box_name_oss;
          box_name_oss << filename << "/box.vtp";
          std::string box_name = box_name_oss.str();

          std::ifstream f(box_name);
          if(!f.good())
            {
              std::ofstream box_vtp;

              Vec3d a1 = xform * Vec3d{domain.origin().x, domain.origin().y, domain.origin().z};
              Vec3d a2 = xform * Vec3d{domain.extent().x, domain.origin().y, domain.origin().z};
              Vec3d a3 = xform * Vec3d{domain.extent().x, domain.extent().y, domain.origin().z};
              Vec3d a4 = xform * Vec3d{domain.origin().x, domain.extent().y, domain.origin().z};
              Vec3d a5 = xform * Vec3d{domain.origin().x, domain.origin().y, domain.extent().z};
              Vec3d a6 = xform * Vec3d{domain.extent().x, domain.origin().y, domain.extent().z};
              Vec3d a7 = xform * Vec3d{domain.extent().x, domain.extent().y, domain.extent().z};
              Vec3d a8 = xform * Vec3d{domain.origin().x, domain.extent().y, domain.extent().z};

              box_vtp.open(box_name);
              box_vtp << "<?xml version=\"1.0\"?>" << std::endl
                      << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" >" << std::endl
                      << vtk_space_offset_two << "<PolyData>" << std::endl
                      << vtk_space_offset_four << "<Piece NumberOfPoints=\"8\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"6\">" << std::endl
                      << vtk_space_offset_six << "<Points>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl
                      << vtk_space_offset_ten
                      << a1.x << " " << a1.y << " " << a1.z << " "
                      << a2.x << " " << a2.y << " " << a2.z << " "
                      << a3.x << " " << a3.y << " " << a3.z << " "
                      << a4.x << " " << a4.y << " " << a4.z << " "
                      << a5.x << " " << a5.y << " " << a5.z << " "
                      << a6.x << " " << a6.y << " " << a6.z << " "
                      << a7.x << " " << a7.y << " " << a7.z << " "
                      << a8.x << " " << a8.y << " " << a8.z << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_six << "</Points>" << std::endl
                      << vtk_space_offset_six << "<Polys>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl
                      << vtk_space_offset_ten << "0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 2 6 5" << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"> " << std::endl
                      << vtk_space_offset_ten << "4 8 12 16 20 24" << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_six << "</Polys>" << std::endl
                      << vtk_space_offset_four << "</Piece>" << std::endl
                      << vtk_space_offset_two << "</PolyData>" << std::endl
                      << "</VTKFile>" << std::endl;

              box_vtp.close();
            }
        }
      //---------------------------------------------------------------------------------------



      //----------------------------------EXTERNAL BOX-----------------------------------------
      // External box is a box taking inside ghosts
      // TO DO : box will be dependant of the time
      if(write_external_box && rank==0)
        {
          ldbg << "write external box" << std::endl;
          std::stringstream box_name_oss;
          box_name_oss << filename << "_external_box.vtp";
          std::string box_name = box_name_oss.str();

          std::ifstream f(box_name);

          size_t glayers = grid.ghost_layers();

          // origin of the box
          // Depend if their are periodics conditions
          double o_x = domain.periodic_boundary_x() ? domain.origin().x - domain.cell_size()*glayers : domain.origin().x;
          double o_y = domain.periodic_boundary_y() ? domain.origin().y - domain.cell_size()*glayers : domain.origin().y;
          double o_z = domain.periodic_boundary_z() ? domain.origin().z - domain.cell_size()*glayers : domain.origin().z;

          // end of the box
          double e_x = domain.periodic_boundary_x() ? domain.extent().x + domain.cell_size()*glayers : domain.extent().x;
          double e_y = domain.periodic_boundary_y() ? domain.extent().y + domain.cell_size()*glayers : domain.extent().y;
          double e_z = domain.periodic_boundary_z() ? domain.extent().z + domain.cell_size()*glayers : domain.extent().z;

          Vec3d a1 = xform * Vec3d{o_x, o_y, o_z};
          Vec3d a2 = xform * Vec3d{e_x, o_y, o_z};
          Vec3d a3 = xform * Vec3d{e_x, e_y, o_z};
          Vec3d a4 = xform * Vec3d{o_x, e_y, o_z};
          Vec3d a5 = xform * Vec3d{o_x, o_y, e_z};
          Vec3d a6 = xform * Vec3d{e_x, o_y, e_z};
          Vec3d a7 = xform * Vec3d{e_x, e_y, e_z};
          Vec3d a8 = xform * Vec3d{o_x, e_y, e_z};

          if(!f.good())
            {
              std::ofstream box_vtp;
              box_vtp.open(box_name);
              box_vtp << "<?xml version=\"1.0\"?>" << std::endl
                      << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" >" << std::endl
                      << vtk_space_offset_two << "<PolyData>" << std::endl
                      << vtk_space_offset_four << "<Piece NumberOfPoints=\"8\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"6\">" << std::endl
                      << vtk_space_offset_six << "<Points>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl
                      << vtk_space_offset_ten
                      << a1.x << " " << a1.y << " " << a1.z << " "
                      << a2.x << " " << a2.y << " " << a2.z << " "
                      << a3.x << " " << a3.y << " " << a3.z << " "
                      << a4.x << " " << a4.y << " " << a4.z << " "
                      << a5.x << " " << a5.y << " " << a5.z << " "
                      << a6.x << " " << a6.y << " " << a6.z << " "
                      << a7.x << " " << a7.y << " " << a7.z << " "
                      << a8.x << " " << a8.y << " " << a8.z << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_six << "</Points>" << std::endl
                      << vtk_space_offset_six << "<Polys>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl
                      << vtk_space_offset_ten << "0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 2 6 5" << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_eight << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"> " << std::endl
                      << vtk_space_offset_ten << "4 8 12 16 20 24" << std::endl
                      << vtk_space_offset_eight << "</DataArray>" << std::endl
                      << vtk_space_offset_six << "</Polys>" << std::endl
                      << vtk_space_offset_four << "</Piece>" << std::endl
                      << vtk_space_offset_two << "</PolyData>" << std::endl
                      << "</VTKFile>" << std::endl;

              box_vtp.close();
            }
        }
      //---------------------------------------------------------------------------------------
      ldbg << "write END" << std::endl;
    }
  }

}
