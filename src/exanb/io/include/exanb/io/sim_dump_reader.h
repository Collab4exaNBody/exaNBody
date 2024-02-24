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

#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <exanb/io/mpi_file_io.h>
#include <exanb/io/sim_dump_io.h>

#include <string>
#include <mpi.h>

namespace exanb
{
  

  template<class LDBG, class GridT, class OptionalDumpFilter, class ... DumpFieldIds>
  static inline void read_dump_transcode(MPI_Comm comm, LDBG& ldbg, GridT& grid, Domain& domain, double & phystime, long & timestep, const std::string& filename, OptionalDumpFilter dump_filter, FieldSet< DumpFieldIds... > dump_fields )
  {
    using RealDumpFields = FieldSetIntersection< typename GridT::Fields , FieldSet< DumpFieldIds... > >;
    using ParticleTuple = onika::soatl::FieldTupleFromFieldIds< RealDumpFields >;
    using StorageTuple = std::remove_cv_t< std::remove_reference_t< decltype( dump_filter.encode( ParticleTuple{} ) ) > >;
    using CompactFieldArrays = compact_field_arrays_from_tuple_t< StorageTuple >;
    
    //static constexpr RealDumpFields real_dump_fields = {};
    //static constexpr size_t READ_BUFFER_SIZE = 262144;

    std::string basename;
    std::string::size_type p = filename.rfind("/");
    if( p != std::string::npos ) basename = filename.substr(p+1);
    else basename=filename;
    
    lout << "============ "<< basename <<" ============" << std::endl ;

    // get MPi configuration
    int np=1, rank=0;
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&rank);

    // open output file
    MpiIO file;
    file.open( comm, filename , "r" );

    // read header from file 
    SimDumpHeader<> header = {};
    file.read( &header );
    file.increment_offset( &header );

    // apply post read filtering, in case version backward compatibility needs some workarounds
    header.post_process();

    if( ! header.check(StorageTuple{}) )
    {
      lerr << "Bad header, abort reading" << std::endl;
      std::abort();
    }
    lout << "fields        =" ;
    StorageTuple{}.apply_fields( print_tuple_field_list(lout) );
    lout <<std::endl;

    // get grid setup
    domain = header.m_domain;
    Mat3d particle_read_xform = make_identity_matrix();
    dump_filter.process_domain( domain , particle_read_xform );

    
    if( ! is_identity(particle_read_xform) )
    {
      lout << "read xform    = "<< particle_read_xform <<std::endl;
    }
    
    lout << "domain bounds = "<<domain.bounds()<<" , size="<<bounds_size(domain.bounds())<<std::endl;
    if( ! domain.xform_is_identity() )
    {
      lout << "domain xform  = "<<domain.xform()<<" , inv = " << domain.inv_xform() << std::endl;
      Vec3d domsize = domain.xform() * bounds_size(domain.bounds());
      lout << "domain size   = "<<domsize << std::endl;
    }
    lout << "cell size     = "<<domain.cell_size()<<std::endl;
    lout << "grid size     = "<<domain.grid_dimension()<<std::endl;
    lout << "periodicity   = "<< std::boolalpha << domain.periodic_boundary_x() << " , " << domain.periodic_boundary_y() << " , "  << domain.periodic_boundary_z() << std::endl;
    lout << "expandable    = "<< std::boolalpha << domain.expandable() << std::endl;
    lout << "particles     = "<< header.m_nb_particles << std::endl;

    phystime = header.m_time;
    timestep = header.m_time_step;
    lout << "time step     = "<< timestep << std::endl;
    lout << "time          = "<< phystime << std::endl;

    ldbg << "opt. header @ = "<< header.m_optional_offset << std::endl;
    lout << "opt. size     = "<< header.m_optional_header_size;
    size_t header_compressed_size = header.m_table_offset - header.m_optional_offset;
    if( header_compressed_size == header.m_optional_header_size ) lout << " (uncompressed)";
    else lout <<" (compressed size "<< header_compressed_size<<")";
    lout << std::endl;
    ldbg << "chunk table @ = "<< header.m_table_offset << std::endl;
    lout << "data chunks   = "<< header.m_chunk_count << std::endl;
    ldbg << "part. data @  = "<< header.m_data_offset << std::endl;

    grid.clear_particles();
    grid.set_cell_size( domain.cell_size() );
    grid.set_max_neighbor_distance( 0.0 );
    grid.set_origin( domain.origin() );
    grid.set_offset( {0,0,0} );
    const IJK grid_dims = domain.grid_dimension();
    grid.set_dimension( grid_dims );

    // read optional header
    ldbg << "file pos = "<< file.current_offset() <<std::endl;

    {
      std::string buffer;
      buffer.resize(header_compressed_size);
      file.read( buffer.data() , header_compressed_size );
      file.increment_offset( buffer.data() , header_compressed_size );
      if( header_compressed_size != header.m_optional_header_size )
      {
        std::string uncompressed_buffer;
        uncompressed_buffer.resize( header.m_optional_header_size );
        MpiIO::uncompress_buffer( buffer.data(), buffer.size(), uncompressed_buffer.data(), uncompressed_buffer.size() );
        buffer = std::move( uncompressed_buffer );
      }
      std::istringstream iss(buffer);
      [[maybe_unused]] size_t optional_header_size = dump_filter.read_optional_header( buffer_read_func(iss) );
      assert( optional_header_size == header.m_optional_header_size );
    }
    ldbg << "optional header size = "<< header.m_optional_header_size << std::endl;
    ldbg << "file pos = "<< file.current_offset() <<std::endl;    

    // read data chunk offset table
    assert( file.current_offset() == static_cast<MPI_Offset>(header.m_table_offset) );
    std::vector<DataChunkItem> chunk_table( header.m_chunk_count );
    file.read( chunk_table.data() , header.m_chunk_count );

    file.increment_offset( chunk_table.data() , header.m_chunk_count );      
    assert( file.user_offset() == static_cast<MPI_Offset>(header.m_data_offset) );
        
    ssize_t first_zero_chunk = -1;
    ssize_t inc_global_ofset = 0;
    for( size_t i = rank ; i < header.m_chunk_count ; i += np )
    {
      //ldbg << "chunk #"<<i<<" : offset="<<chunk_table[i].m_global_offset<<", size="<<chunk_table[i].m_data_size<<", particles="<<std::abs(chunk_table[i].m_n_particles)<<", ext="<<std::boolalpha<<(chunk_table[i].m_n_particles<0)<<std::endl;
      if( !chunk_table[i].valid() || static_cast<ssize_t>(chunk_table[i].m_global_offset) < inc_global_ofset )
      {
        lerr << "file corrupted, inconsistent chunk #"<<i<<std::endl;
        std::abort();
      }
      else
      {
        inc_global_ofset = static_cast<ssize_t>(chunk_table[i].m_global_offset);
      }

      if( chunk_table[i].empty() )
      {
        if( first_zero_chunk == -1 ) first_zero_chunk = i;
      }
      else
      {
        assert( first_zero_chunk == -1 ); // sequence of zero chunks not at end => error
      }
    }
    if( first_zero_chunk != -1 )
    {
      ldbg << "first zero chunk encountered : "<< first_zero_chunk << std::endl;
    }

    unsigned long long particle_count = 0;
    size_t at_id = 0;
    size_t n_zero_in = 0;
    size_t n_zero = 0;
    ssize_t zero_seq_start = -1;
    ssize_t zero_seq_end = -1;
    
    // read particles
    dump_filter.initialize_read();
    CompactFieldArrays particle_buffer;
    std::string compressed_buffer;
    std::vector<uint8_t> optional_data_buffer;
    std::vector<uint8_t> uncompress_buffer;

    // lout.progress_bar("Loading ",0.0);
    for( size_t i = rank ; i < header.m_chunk_count ; i += np ) if( ! chunk_table[i].empty() )
    {
      assert( static_cast<MPI_Offset>(chunk_table[i].m_global_offset) >= file.user_offset() );
      file.increment_user_offset_bytes( chunk_table[i].m_global_offset - file.user_offset() );
      assert( std::abs(chunk_table[i].m_n_particles) > 0 );
      const size_t n_particles = std::abs(chunk_table[i].m_n_particles);
      const size_t guessed_compressed_chunk_size = ( (i+1)<chunk_table.size() ) ? ( chunk_table[i+1].m_global_offset - chunk_table[i].m_global_offset ) : 0;
      
      file.collective_update_file_parts();

      optional_data_buffer.clear();
      particle_buffer.clear();
      particle_buffer.resize( n_particles );
      particle_buffer.shrink_to_fit();
      
      if( chunk_table[i].m_n_particles > 0 ) // chunk[i].m_data_size means compressed buffer size, raw size is gussed from number of particles, no optional data
      {
        const size_t uncompressed_buffer_size = particle_buffer.storage_size();
        const size_t compressed_buffer_size = chunk_table[i].m_data_size;
        assert( guessed_compressed_chunk_size==0 || guessed_compressed_chunk_size==compressed_buffer_size );
        if( compressed_buffer_size == uncompressed_buffer_size ) // uncompressed storage detected
        {
          file.read_bytes( particle_buffer.storage_ptr() , uncompressed_buffer_size );
        }
        else
        {
          compressed_buffer.resize( compressed_buffer_size );
          file.read( compressed_buffer.data() , compressed_buffer_size );
          MpiIO::uncompress_buffer( compressed_buffer.data(), compressed_buffer_size, (char*) particle_buffer.storage_ptr() , uncompressed_buffer_size );
        }
        //ldbg << "read "<<n_particles<<" from chunk "<< i<<", size = "<< uncompressed_buffer_size <<" / "<< compressed_buffer_size << " (no optional data)" << std::endl;
      }
      else
      {
        if( guessed_compressed_chunk_size == 0 )
        {
          lerr << "impossible to read chunk"<<i<<", chunk table corrupted (probably missing last empty chunk)\n";
          std::abort();          
        }

        const size_t uncompressed_buffer_size = chunk_table[i].m_data_size;
        const size_t compressed_buffer_size = guessed_compressed_chunk_size;

        if( compressed_buffer_size == uncompressed_buffer_size )
        {
          assert( uncompressed_buffer_size > particle_buffer.storage_size() );
          file.read_bytes( particle_buffer.storage_ptr() , particle_buffer.storage_size() );
          file.increment_user_offset_bytes( particle_buffer.storage_size() );
          optional_data_buffer.resize( uncompressed_buffer_size - particle_buffer.storage_size() );
          file.read_bytes( optional_data_buffer.data() , optional_data_buffer.size() );
        }
        else
        {
          compressed_buffer.resize( compressed_buffer_size );
          file.read( compressed_buffer.data() , compressed_buffer_size );
          uncompress_buffer.resize( uncompressed_buffer_size );
          MpiIO::uncompress_buffer( compressed_buffer.data(), compressed_buffer_size, (char*) uncompress_buffer.data() , uncompressed_buffer_size );
          std::memcpy( particle_buffer.storage_ptr() , uncompress_buffer.data() , particle_buffer.storage_size() );
          optional_data_buffer.resize( uncompressed_buffer_size - particle_buffer.storage_size() );
          std::memcpy( optional_data_buffer.data() , uncompress_buffer.data() + particle_buffer.storage_size() , optional_data_buffer.size() );
          uncompress_buffer.clear();
        }
        //ldbg << "read "<<n_particles<<" from chunk "<< i<<", size = "<< uncompressed_buffer_size <<" (opt. "<<optional_data_buffer.size()<<") / "<< compressed_buffer_size << std::endl;
        dump_filter.read_optional_data_from_stream( optional_data_buffer.data() , optional_data_buffer.size() );
        optional_data_buffer.clear();
      }
      
      for(size_t i=0;i<n_particles;i++)
      {
        ParticleTuple tp = dump_filter.decode( particle_buffer[i] );
        Vec3d ro = particle_read_xform * Vec3d{ tp[field::rx] , tp[field::ry] , tp[field::rz] };
        Vec3d r = ro;
        IJK loc = domain_periodic_location( domain, r ) - grid.offset();

        // optionally filter out some particles at read time
        if( dump_filter.particle_input_filter( domain.xform() * r ) )
        {
          if( ! grid.contains(loc) )
          {
            IJK badloc = loc;
            if( loc.i < 0 ) loc.i = 0;
            else if( loc.i >= domain.grid_dimension().i ) loc.i = domain.grid_dimension().i - 1;
            if( loc.j < 0 ) loc.j = 0;
            else if( loc.j >= domain.grid_dimension().j ) loc.j = domain.grid_dimension().j - 1;
            if( loc.k < 0 ) loc.k = 0;
            else if( loc.k >= domain.grid_dimension().k ) loc.k = domain.grid_dimension().k - 1;
            
            double dist = min_distance_between( r , grid.cell_bounds(loc) );
            if( dist < grid.cell_size()*1.e-3 )
            {
              lerr << "Warning: outside particle, location adjusted from "<<badloc<<" to "<<loc<<", distance="<<dist<<std::endl;
            }
            else
            {
              lerr<<"Domain = "<<domain<<std::endl;
              lerr<<"Domain size = "<<domain.bounds_size()<<std::endl;
              lerr<<"particle #"<<at_id<<", ro="<<ro<<", r="<<r<<"<< in cell "<<loc<<" (dist="<<dist <<") not in grid : offset="<<grid.offset()<<std::endl<<std::flush;
              std::abort();
            }
          }

          if( ro == Vec3d{0,0,0} )
          {
            ++ n_zero_in;
            if( zero_seq_start == -1 ) { zero_seq_start=at_id; zero_seq_end=at_id+1; }
            else if( ssize_t(at_id) == zero_seq_end ) { ++ zero_seq_end; }
          }
          else if( zero_seq_end>zero_seq_start )
          {
            ldbg << "particles #"<<zero_seq_start<<"-"<<(zero_seq_end-1)<<" ("<<zero_seq_end-zero_seq_start<<") are located at (0,0,0)" << std::endl;
            zero_seq_start = -1;
            zero_seq_end = -1;
          }
          
          if( r == Vec3d{0,0,0} ) { ++ n_zero; }

          // update particle's position with respect to boundary conditions
          tp[field::rx] = r.x;
          tp[field::ry] = r.y;
          tp[field::rz] = r.z;

          const size_t cell_idx = grid_ijk_to_index( grid_dims , loc );
          auto & cell = grid.cell(cell_idx);
          const size_t p_idx = cell.size();
          assert( is_inside( grid.cell_bounds(loc) , Vec3d{tp[field::rx],tp[field::ry],tp[field::rz]} ) );
          cell.push_back( tp );
          dump_filter.append_cell_particle( cell_idx , p_idx );
          ++ at_id;
          ++ particle_count;          
        }
      }

      // lout.progress_bar("Loading ",(i+1)*1.0/local_chunk_count);
    }            

    // free resources
    particle_buffer.clear();
    particle_buffer.shrink_to_fit();
    
    dump_filter.finalize_read();

    while( file.collective_update_file_parts(false) ); // keep participating until all prrocesses are done
    file.close();

    if( zero_seq_end>zero_seq_start )
    {
      ldbg << "particles #"<<zero_seq_start<<"-"<<(zero_seq_end-1)<<" are located at (0,0,0)" << std::endl;
      zero_seq_start = -1;
      zero_seq_end = -1;
    }

    if( n_zero_in > 1 )
    {
      lerr << n_zero_in<<" particles at position (0,0,0) in input file\n";
      std::abort();
    }
    if( n_zero > 1 )
    {
      lerr << "Internal error: "<<n_zero <<" particles at position (0,0,0) in grid\n";
      std::abort();
    }

    dump_filter.post_process_domain( domain );
    IJK cell_shift = make_ijk( floor( ( domain.origin() - grid.origin() ) / domain.cell_size() ) );
    grid.set_origin( grid.origin() + cell_shift * domain.cell_size() );
    grid.set_offset( grid.offset() - cell_shift );
    grid.rebuild_particle_offsets();
    assert( check_particles_inside_cell(grid/*,true,true*/) );
    ldbg << "grid has "<< grid.number_of_particles() << " particles" << std::endl; 
    
    MPI_Allreduce(MPI_IN_PLACE,&particle_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    assert( particle_count == header.m_nb_particles );
    ldbg << "total particles read = " << particle_count << std::endl ;
    
    lout << "================================" << std::endl << std::endl;
  }

  template<class LDBG, class GridT, class DumpFieldSet, class OptionalDumpFilter = NullDumpOptionalFilter >
  static inline void read_dump(MPI_Comm comm, LDBG& ldbg, GridT& grid, Domain& domain, double & phystime, long & timestep, const std::string& filename, DumpFieldSet dump_fields, OptionalDumpFilter dump_filter = {} )
  {
    read_dump_transcode( comm, ldbg, grid, domain, phystime, timestep, filename, dump_filter, dump_fields );
  }

  template<typename GridT, class FS, class OptionalDumpFilter = NullDumpOptionalFilter > class SimDumpReader;

  template<typename GridT, class OptionalDumpFilter, class ... DumpFieldIds >
  class SimDumpReader< GridT , FieldSet<DumpFieldIds...> , OptionalDumpFilter > : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT       , grid     , INPUT_OUTPUT );
    ADD_SLOT( Domain      , domain   , INPUT_OUTPUT );
    ADD_SLOT(long         , timestep      , INPUT_OUTPUT , DocString{"Iteration number"} );
    ADD_SLOT(double       , physical_time , INPUT_OUTPUT , DocString{"Physical time"} );
    ADD_SLOT( std::string , filename , INPUT );

  public:
    inline void execute () override final
    {
      read_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, FieldSet< DumpFieldIds... >{} , OptionalDumpFilter{} );
    }
  };

}


