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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/oarray.h>

#include <exanb/io/vtk_writer.h>
#include <exanb/compute/field_combiners.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include <string>
#include <type_traits>
#include <experimental/filesystem>
#include <regex>
#include <filesystem>

namespace exanb
{

  namespace write_xyz_details
  {

    template<class T,size_t N> struct IsArrayOfInteger : public std::false_type {};
    template<class T,size_t N> struct IsArrayOfInteger< std::array<T,N> , N > : public std::is_integral<T> {};
    template<class T,size_t N> struct IsArrayOfInteger< onika::oarray_t<T,N> , N > : public std::is_integral<T> {};
    template<class T,size_t N> static inline constexpr bool is_array_of_integral_v = IsArrayOfInteger<T,N>::value ;
    
    struct DefaultFieldFormatter
    {
      std::map<std::string,std::string> m_field_unit;
      std::unordered_map<std::string,double> m_conv_map;
      std::map<std::string,std::string> m_field_name_map;
      
      struct no_field_id_t { static inline const char* short_name() { return "no-field-id"; } };
   
      template<class T>
      inline std::string field_name_mapping(const T& f) const
      {
        auto it = m_field_name_map.find(f.short_name());
        if( it != m_field_name_map.end() ) return it->second;
        else return f.short_name();
      }

      template<class T>
      static inline const std::string& format_for_value (const T& f)
      {
        static const std::string integer = "% 10d";
        static const std::string int4 = "% 10d % 10d % 10d % 10d";
        static const std::string real = "% .10e";
        static const std::string real3 = "% .10e % .10e % .10e";
        static const std::string real9 = "% .10e % .10e % .10e % .10e % .10e % .10e % .10e % .10e % .10e";
        static const std::string other = "%-8s";
        using field_type =std::remove_cv_t< std::remove_reference_t<T> >;
        if constexpr ( ParaViewTypeId<field_type>::ncomp == 1 )
        {
          if ( std::is_integral_v<field_type> ) return integer;
          else if ( std::is_arithmetic_v<field_type> ) return real;
          else return other;
        }
        else if constexpr ( std::is_same_v<field_type,Vec3d> )
        {
          return real3;
        }
        else if constexpr ( std::is_same_v<field_type,Mat3d> )
        {
          return real9;
        }
        else if constexpr ( is_array_of_integral_v<field_type,4> )
        {
          return int4;
        }
        return other;
      }
      
      template<class FieldIdT>
      inline std::string property_for_field (const FieldIdT& f) const
      {
        std::string fname = field_name_mapping(f);
        using field_type =std::remove_cv_t< std::remove_reference_t<typename FieldIdT::value_type> >;      
        if constexpr ( std::is_same_v<field_type,Vec3d> )
        {
          return fname + ":R:3";
        }
        else if constexpr ( std::is_same_v<field_type,Mat3d> )
        {
          return fname + ":R:9";
        }
        else if constexpr ( std::is_same_v<field_type,Quaternion> )
        {
          return fname + ":R:4";
        }
        else if constexpr ( is_array_of_integral_v<field_type,4> )
        {
          return fname + ":I:4";
        }
        else if constexpr ( std::is_integral_v<field_type> )
        {
          return fname + ":I:1";
        }
        else if constexpr ( std::is_arithmetic_v<field_type> )
        {
          return fname + ":R:1";
        }
	//else // commented out to avoid intel compiler fake warning about missing return value
	//{
          return fname + ":S:1";
	//}
      }
      
      template<class T, class FieldIdT = no_field_id_t >
      inline int operator () (char* buf, int bufsize, const T& in_v , FieldIdT f = {} ) const
      {
        static const std::string field_short_name( f.short_name() );
        double conv = 1.0;
        if constexpr ( ! std::is_same_v<FieldIdT,no_field_id_t> )
        {
          auto it = m_conv_map.find( f.short_name() );
          if( it != m_conv_map.end() ) conv = it->second;
        }
        using field_type =std::remove_cv_t< std::remove_reference_t<T> >; 
        if( bufsize < 0 ) return 0;
        else if constexpr ( std::is_same_v<field_type,Vec3d> )
        {
          const T v = ( conv != 1.0 ) ? static_cast<T>( in_v * conv ) : in_v;
          return format_string_buffer( buf, bufsize, format_for_value(v) , v.x , v.y , v.z );
        }
        else if constexpr ( std::is_same_v<field_type,Mat3d> )
        {
          const T v = ( conv != 1.0 ) ? static_cast<T>( in_v * conv ) : in_v;
          return format_string_buffer( buf, bufsize, format_for_value(v) , v.m11 , v.m12 , v.m13 , v.m21 , v.m22 , v.m23 , v.m31 , v.m32 , v.m33 );
        }
        else if constexpr ( std::is_same_v<field_type,Quaternion> )
        {
          const T v = ( conv != 1.0 ) ? static_cast<T>( in_v * conv ) : in_v;
          return format_string_buffer( buf, bufsize, format_for_value(v) , v.w , v.x , v.y , v.z );
        }
        else if constexpr ( is_array_of_integral_v<field_type,4> )
        {
          T v; for(size_t i=0;i<4;i++) v[i] = ( conv != 1.0 ) ? static_cast<decltype(in_v[i])>( in_v[i] * conv ) : in_v[i] ;
          return format_string_buffer( buf, bufsize, format_for_value(v) , v[0] , v[1] , v[2] , v[3] );
        }
        else if constexpr ( std::is_arithmetic_v<field_type> )
        {
          const T v = ( conv != 1.0 ) ? static_cast<T>( in_v * conv ) : in_v;
          return format_string_buffer( buf, bufsize, format_for_value(v) , v );
        }
        else // commented out to avoid intel compiler fake warning about missing return value
        {
          if( ParaViewTypeId<field_type>::ncomp != 1 ) { fatal_error() << "number of components not 1 as expected for field "<<f.short_name()<<" with type "<<type_as_string<T>()<<std::endl; }
          if( conv != 1.0 ) { fatal_error() << "Conversion factor not allowed for type "<<typeid(T).name()<<std::endl; }
          return format_string_buffer( buf, bufsize, format_for_value(in_v) , in_v );
        }
	return 0; // should never get there
      }

    };

    struct NullParticleTypleStrFunctor
    {
      template<class CellsT>
      inline const char* operator () (CellsT cells, size_t cell, size_t particle) const
      {
        return "XX";
      }
    };

    template<class LDBGT, class GridT,  class ParticleTypeStrFuncT, class FieldFormatterT, class... FieldsT>
    static inline void write_xyz_grid_fields(
      LDBGT& ldbg,
      MPI_Comm comm, 
      const GridT& grid, 
      const Domain& domain, 
      const std::vector<std::string>& flist, 
      const std::string& filename, 
      const ParticleTypeStrFuncT& particle_type_func,
      const FieldFormatterT& formatter ,
      bool write_ghosts,
      double time,
      const FieldsT & ... particle_fields)
    {
      namespace fs = std::experimental::filesystem;
      using std::string;
      using std::vector;
      using std::ostringstream;
      using namespace write_xyz_details;

      Mat3d xform = domain.xform();
            
      size_t n_cells = grid.number_of_cells();
      auto field_selector = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;
      auto cells = grid.cells_accessor();
      
      unsigned long nb_particles = grid.number_of_particles();
      if( ! write_ghosts ) { nb_particles -= grid.number_of_ghost_particles(); }
      
      int rank=0, np=1;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &np);

      ostringstream filename_xyz;
      filename_xyz << std::string(filename.c_str());

      // we don't want a proc try to write in a folder that doesn't exist
      MPI_Barrier(comm);

      if (rank==0)
        {
          std::filesystem::path dir_path = std::filesystem::path(filename).parent_path();
          if( dir_path != "" )
          {
            std::filesystem::create_directories( dir_path );
          }
        }

      MPI_Barrier(comm);
      
      // structure for file opening/writing in mpi
      MPI_File mpiFile;
      MPI_Status status;

      // all the processors open the .xyz file
      MPI_File_open(MPI_COMM_WORLD, filename_xyz.str().c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpiFile);

      string header;
      unsigned long  nb_particles_offset=0;

      // compute the sum of particle of above processors me included
      MPI_Scan(&nb_particles, &nb_particles_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

      // retrieve my number of particles to compute the right offset
      nb_particles_offset = nb_particles_offset - nb_particles;

      // count total number of particles among all processors
      unsigned long total_particle_number = 0;
      for(size_t c=0; c<n_cells;++c)
      {
        int np = 0;
        if( !grid.is_ghost_cell(c) || write_ghosts )
        {
          np = cells[c].size();
        }
        total_particle_number += np;
      }
      MPI_Allreduce(MPI_IN_PLACE,&total_particle_number,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);

      //matrix defining the shape of the box. It is written in the header of the xyz file
      Mat3d lattice = diag_matrix( domain.extent() - domain.origin() );
      Mat3d lot = transpose(xform * lattice);

      std::ostringstream oss;
      oss << format_string("%ld\nLattice=\"%10.12e %10.12e %10.12e %10.12e %10.12e %10.12e %10.12e %10.12e %10.12e\"",total_particle_number, lot.m11, lot.m12, lot.m13, lot.m21, lot.m22, lot.m23, lot.m31, lot.m32, lot.m33);
      // ( ... , (
      //   field_selector(particle_fields.short_name()) ? ( oss << ' ' << formatter.field_name_mapping(particle_fields) ) : oss
      // ) ;
      oss<<" Properties=species:S:1";
      ( ... , (
        field_selector(particle_fields.short_name()) ? ( oss << ':' << formatter.property_for_field(particle_fields) ) : oss
      ) );
      oss << " Time="<<time<<"\n";
      std::string header_data = oss.str();
      size_t offset_header = header_data.length();
  
      // only processor 0 writes the header
      if (rank==0)
      {
        MPI_File_write(mpiFile, header_data.data() , header_data.length() , MPI_CHAR , &status);
      }

      auto write_field_to_buf = [&] (char* buf, int capacity, auto f, size_t c, size_t p) -> int
      {
        int n = 0;
        if( field_selector(f.short_name()) )
        {
          if( capacity == 0 )
          {
            ldbg << "buffer full, can't write field "<<f.short_name()<<std::endl;
            return 0;
          }
          *buf = ' ';
          n = 1 + formatter( buf + 1 , capacity - 1 , cells[c][f][p] );
          if( capacity < n )
          {
            fatal_error() << "unsexpcted buffer overflow" << std::endl;
          }
          //ldbg << "write field "<<f.short_name()<<" using 1+"<<n<<" bytes"<<std::endl;
        }
        return n;
      };

      int data_line_size = 0;
      size_t nb_particles_written = 0;

      auto write_position_and_fields = [&] (auto position_field, auto ... f)
      {
        // account for  particle type, then 1 space character, then particle position
        data_line_size = formatter(nullptr,0,std::string("XX")) + 1 + formatter(nullptr,0,decltype(cells[0][position_field][0]){});
        ldbg << "particle data start size = "<<data_line_size<<std::endl;
        ( ... , ( data_line_size += field_selector(f.short_name()) ? ( 1 + formatter(nullptr,0, typename decltype(f)::value_type {} ) ) : 0 ) );
        ++ data_line_size; // we'll add a '\n' at the end of each line      
        ldbg << "particle data total size = "<<data_line_size<<std::endl;
        std::string line_data( data_line_size , ' ' );

        nb_particles_written = 0;
        // routine d'écriture MPI des particules dans le fichier .xyz
        for(size_t c=0; c<n_cells;++c)
        {
          if( !grid.is_ghost_cell(c) || write_ghosts )
          {
            int np = cells[c].size();
            for(int pos=0;pos<np;++pos)
            {
              Vec3d pos_vec = cells[c][position_field][pos];
              auto type_name = particle_type_func ( cells, c, pos );
              pos_vec = xform * pos_vec;

              line_data.assign( data_line_size , ' ');
              char* buf = line_data.data();
              int written = 0;
              written += formatter( buf+written , data_line_size + 1 - written , type_name );
              buf[written++] = ' ';
              written += formatter( buf+written , data_line_size + 1 - written , pos_vec );
              //buf[written] = '\0';
              //ldbg << "particle data start = '"<<line_data.data()<<"' , bytes="<< written<<" , length="<< line_data.length() <<std::endl;              
              ( ... , ( written += write_field_to_buf(buf+written,data_line_size+1-written,f,c,pos) ) );
              //buf[written] = '\0';
              //ldbg << "particle data line = '"<<line_data.data()<<"' , bytes="<< written<<" , length="<< line_data.length() <<std::endl;
              assert( written == (data_line_size-1) );
              buf[written++] = '\n';
              assert( written == data_line_size );
              assert( line_data.length() == size_t(data_line_size) );
              //ldbg << "particle data final = '"<<line_data.data()<<"' , bytes="<< written<<" , length="<< line_data.length() <<std::endl;              
              size_t offset = offset_header + ( nb_particles_offset + nb_particles_written ) * data_line_size;
              ++ nb_particles_written;
              MPI_File_write_at( mpiFile, offset, line_data.data(), data_line_size , MPI_CHAR , &status );
            }
          }
        }
      };      
      
      write_position_and_fields( particle_fields ... );
      
      MPI_Barrier(comm);
      MPI_File_close(&mpiFile);
    }

  } // end of write_xyz_details namespace
  
  template<class GridT , class ParticleTypeStrFuncT = write_xyz_details::NullParticleTypleStrFunctor , class FieldFormatterT = write_xyz_details::DefaultFieldFormatter >
  class WriteXYZGeneric : public OperatorNode
  {    
    using StringList = std::vector<std::string>;
        
    ADD_SLOT( MPI_Comm             , mpi      , INPUT );
    ADD_SLOT( GridT                , grid     , INPUT );
    ADD_SLOT( Domain               , domain   , INPUT );
    ADD_SLOT( bool                 , ghost    , INPUT , false );
    ADD_SLOT( std::string          , filename , INPUT , "output"); // default value for backward compatibility
    ADD_SLOT( StringList           , fields   , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to project"} );
    ADD_SLOT( ParticleTypeStrFuncT , particle_type_func , INPUT , ParticleTypeStrFuncT{} );
    ADD_SLOT( FieldFormatterT      , field_formatter , INPUT , FieldFormatterT{} );
      
    template<class... fid>
    inline void execute_on_field_set( FieldSet<fid...> ) 
    {
      int rank=0;
      MPI_Comm_rank(*mpi, &rank);
      ProcessorRankCombiner processor_id = { {rank} };
      
      PositionVec3Combiner position = {};
      VelocityVec3Combiner velocity = {};
      ForceVec3Combiner    force    = {};

      // property name for position must be 'Position'
      StringList flist = { "position" };
      for(const auto& f : *fields) { if( f != "position" ) flist.push_back(f); }

      auto formatter = *field_formatter;
      formatter.m_field_name_map["position"] = "pos";

      write_xyz_details::write_xyz_grid_fields( ldbg, *mpi, *grid, *domain, flist, *filename, *particle_type_func, formatter, *ghost, 0.0
                                              , position, velocity, force, processor_id, onika::soatl::FieldId<fid>{} ... );
    }

    public:
    inline void execute() override
    {
      using GridFieldSet = RemoveFields< typename GridT::field_set_t , FieldSet< field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz> >;
      execute_on_field_set( GridFieldSet{} );
    }
    
  };

}

