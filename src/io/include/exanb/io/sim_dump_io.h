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

#include <exanb/fields.h>

#include <exanb/core/domain.h>
#include <onika/log.h>
#include <exanb/io/domain_legacy_v1.2.h>

#include <onika/soatl/field_tuple.h>
#include <onika/soatl/field_arrays.h>

#include <string>
#include <set>

namespace exanb
{
  

  struct CompressionBuffer
  {
    std::string m_data;
    uint64_t m_raw_size = 0;
    int64_t m_n_particles = 0;
  };

  struct DataChunkItem
  {
    uint64_t m_global_offset = 0;
    uint32_t m_data_size = 0;
    int32_t m_n_particles = 0;
    inline bool empty() const { return m_data_size==0 && m_n_particles==0; }
    inline bool valid() const { return empty() || ( m_n_particles!=0 && m_data_size!=0 ); }
  };

  template<class ftp> struct CompactFieldArraysFromTuple;
  template<class... fids> struct CompactFieldArraysFromTuple< onika::soatl::FieldTuple<fids...> > { using type = onika::soatl::FieldArrays<8,1,fids...>; };
  template<class ftp> using compact_field_arrays_from_tuple_t = typename CompactFieldArraysFromTuple<ftp>::type;
  
  struct NullDumpOptionalFilter
  {
    template<class TupleT> static inline const TupleT& encode( const TupleT& tp ) { return tp; }
    template<class TupleT> static inline const TupleT& decode( const TupleT& tp ) { return tp; }
    template<class WriteFuncT> static inline constexpr size_t write_optional_header( WriteFuncT ) { return 0; }
    template<class ReadFuncT> static inline constexpr size_t read_optional_header( ReadFuncT ) { return 0; }
    static inline constexpr void process_domain(Domain&,Mat3d&) {}
    static inline constexpr bool particle_input_filter(const Vec3d&) { return true; } // returns false if particle is to be ignored
    static inline constexpr void post_process_domain(Domain&) {}
    static inline constexpr size_t optional_cell_data_size(size_t) { return 0; }
    static inline constexpr void write_optional_cell_data(uint8_t*, size_t) { }
    static inline constexpr void read_optional_data_from_stream( const uint8_t* , size_t ) {}
    static inline constexpr void append_cell_particle( size_t, size_t ) {};
    static inline constexpr void initialize_write() {}
    static inline constexpr void initialize_read() {}
    static inline constexpr void finalize_write() {}
    static inline constexpr void finalize_read() {}
  };

  template<class StreamT>
  struct PrintTupleField
  {
    StreamT& out;
    template<class F,class T> inline void operator () ( F , T ) { std::cout<<"\t"<< F::short_name() << " ("<<sizeof(T)<<")\n"; }
  };
  template<class StreamT> PrintTupleField<StreamT> print_tuple_field(StreamT& out) { return {out}; }

  template<class StreamT>
  struct PrintTupleFieldList
  {
    StreamT& out;
    template<class F,class T> inline void operator () ( F , T ) { out<<" "<< F::short_name() ; }
  };
  template<class StreamT> PrintTupleFieldList<StreamT> print_tuple_field_list(StreamT& out) { return {out}; }

  // helper functor to enumerate possible fields
  struct GatherTupleFields
  {
    std::set<std::string>& stp_fields;
    template<class F,class T>
    inline void operator () ( F , T )
    {
      stp_fields.insert( F::short_name() );
    }
  };


  // version history
  // 1.1 -> 1.2 : force domain.m_expandable to true, only if version is <= 1.1, as it was previously implicitly true, but stored as false (see extend_domain for more information)
  // 1.2 -> 1.3 : Domain structure changed in commit # 3f4b5697a70e263b6fd24f807515c3f2a7c3423b which converted bool to bit mask for periodic, expandable and mirroring flags
# define XNB_IO_MAKE_VERSION_NUMBER(major,minor) ((major)*1000+(minor))
  
  struct SimDumpHeader
  {
    static_assert( sizeof(Domain) == sizeof(Domain_legacy_v1_2) );

    static constexpr uint64_t VERSION = XNB_IO_MAKE_VERSION_NUMBER(1,3);
    static constexpr unsigned int MAX_FIELDS = 128;
    static constexpr unsigned int STR_MAX_LEN = 32;
    
    static constexpr unsigned int DATA_FLAG_UNUSED0 = 0; // index in m_data_flags
    static constexpr unsigned int DATA_FLAG_UNUSED1 = 1; // index in m_data_flags
    static constexpr unsigned int DATA_FLAG_UNUSED2 = 2; // index in m_data_flags
    static constexpr unsigned int DATA_FLAG_UNUSED3 = 3; // index in m_data_flags

    const uint64_t m_version = VERSION;

    uint32_t m_nb_fields = 0;
    uint8_t m_data_flags[4] = { 0 , 0 , 0 , 0 };
    uint64_t m_tuple_size = 0;
    uint64_t m_field_size[MAX_FIELDS]; // byte size per element    
    char m_fields[MAX_FIELDS][STR_MAX_LEN]; // null terminated strings array

    uint64_t m_nb_particles = 0;
    uint64_t m_time_step = 0;
    double m_time = 0.0;

    Domain m_domain;
    
    uint64_t m_optional_offset = 0;
    uint64_t m_table_offset = 0;
    uint64_t m_data_offset = 0;

    uint64_t m_optional_header_size = 0;
    uint64_t m_chunk_count = 0;

    static inline uint8_t compression_threads_encode(unsigned int compression_threads)
    {
      if( compression_threads > 1023 ) compression_threads = 1023;
      return (compression_threads/4);
    }
    
    static inline unsigned int compression_threads_decode(uint8_t comp_flag)
    {
      return (comp_flag==0) ? 1 : (comp_flag*4);
    }

    // called after structure has been read from file
    inline void post_process()
    {
      if( m_version < XNB_IO_MAKE_VERSION_NUMBER(1,3) )
      {
        // data that has been read are actually the content of Domain_legacy_v1_2 struct
        const Domain_legacy_v1_2 * legacy_domain_p = reinterpret_cast<const Domain_legacy_v1_2*>( & m_domain );
        const Domain_legacy_v1_2 legacy_domain = * legacy_domain_p;
        m_domain.set_bounds( legacy_domain.bounds() );
        m_domain.set_grid_dimension( legacy_domain.grid_dimension() );
        m_domain.set_cell_size( legacy_domain.cell_size() );
        m_domain.set_xform( legacy_domain.xform() );
        m_domain.set_mirror_x_min( false );
        m_domain.set_mirror_x_max( false );
        m_domain.set_mirror_y_min( false );
        m_domain.set_mirror_y_max( false );
        m_domain.set_mirror_z_min( false );
        m_domain.set_mirror_z_max( false );
        m_domain.set_periodic_boundary_x( legacy_domain.periodic_boundary_x() );
        m_domain.set_periodic_boundary_y( legacy_domain.periodic_boundary_y() );
        m_domain.set_periodic_boundary_z( legacy_domain.periodic_boundary_z() );
        m_domain.set_expandable( legacy_domain.expandable() );
      }
      if( m_version < XNB_IO_MAKE_VERSION_NUMBER(1,2) )
      {
        if( !m_domain.expandable() )
        {
          m_domain.set_expandable( true );
          ldbg << "Warning: domain/expandable has been forced to true for backward compatibility"<<std::endl;
        }
      }
    }

    template<class StorageTuple>
    inline bool check( StorageTuple ) const
    {
      if( m_version > VERSION )
      {
        lerr<<"SimDumpHeader::check : bad version number "<<m_version/1000<<'.'<<m_version%1000<<" , expected "<<VERSION/1000<<'.'<<VERSION%1000<<" or less"<< std::endl;
        return false;
      }
      
      if( sizeof(StorageTuple) != m_tuple_size )
      {
        lerr<<"SimDumpHeader::check : bad tuple size : got "<<m_tuple_size<<", but expected "<<sizeof(StorageTuple)<<std::endl;
        return false;
      }
      
      std::set<std::string> io_fields;
      for(uint64_t i=0;i<m_nb_fields;i++)
      {
        io_fields.insert( m_fields[i] );
      }
      std::set<std::string> stp_fields;
      StorageTuple{}.apply_fields( GatherTupleFields{stp_fields} );
      if( io_fields != stp_fields )
      {
        lerr<<"SimDumpHeader::check : field set mismatch"<<std::endl<<"header contains"<<std::endl;
        for(const auto& s:io_fields) lerr<<"\t"<<s<<std::endl;
        lerr<<"template expects"<<std::endl;
        for(const auto& s:stp_fields) lerr<<"\t"<<s<<std::endl;
        return false;
      }
      return true;
    }
  };

}


