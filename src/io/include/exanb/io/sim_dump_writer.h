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

#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>

#include <exanb/io/mpi_file_io.h>
#include <exanb/io/sim_dump_io.h>

#include <string>
#include <limits>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <filesystem>

namespace exanb
{
  
  struct SimDumpInitializer
  {
    SimDumpHeader & m_header;

    template<class FieldId,class T>
    inline void operator () ( FieldId , T )
    {
        static constexpr size_t field_size = sizeof( typename FieldId::value_type );
        assert( m_header.m_nb_fields < m_header.MAX_FIELDS );
        std::strncpy( m_header.m_fields[ m_header.m_nb_fields ] , FieldId::short_name() , m_header.STR_MAX_LEN-1 );
        m_header.m_fields[ m_header.m_nb_fields ][m_header.STR_MAX_LEN-1] = '\0';
        m_header.m_field_size[ m_header.m_nb_fields ] = field_size ;
        ++ m_header.m_nb_fields;
    }
  };

  inline void collective_write_offset(MPI_Comm comm, long np, unsigned long long cur_offset, unsigned long long bytes, unsigned long long& write_offset, unsigned long long& next_offset)
  {
    write_offset = 0;
    MPI_Scan( &bytes , &write_offset , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , comm );
    write_offset = cur_offset + write_offset - bytes;
    next_offset = write_offset + bytes;
    MPI_Bcast( &next_offset , 1 , MPI_UNSIGNED_LONG_LONG , np-1 , comm );        
  }

  // may be used with buffers={} to allow other processes to progress without writing anything
  inline unsigned long long collective_write(MpiIO& file, MPI_Comm comm, long np, unsigned long long cur_offset, std::vector<DataChunkItem>& chunk_table, const std::vector<CompressionBuffer>& buffers )
  {
    unsigned long long write_offset = 0;
    unsigned long long next_offset = 0;

    size_t total_size = 0;
    for(const auto& b:buffers) total_size += b.m_data.size();

    collective_write_offset( comm, np, cur_offset, total_size , write_offset , next_offset );
    assert( static_cast<MPI_Offset>(write_offset) >= file.user_offset() );

    for(const auto& b:buffers)
    {
      assert( static_cast<MPI_Offset>(write_offset) >= file.user_offset() );
      if( ! b.m_data.empty() )
      {
        // assert( b.m_n_particles > 0 ); // NOT NECESSARY : m_n_particles < 0 tells not to store compressed size
        assert( b.m_raw_size >= b.m_data.size() );
        assert( b.m_data.size() < std::numeric_limits<uint32_t>::max() );
        assert( b.m_raw_size < std::numeric_limits<uint32_t>::max() );
        const bool store_comp_size = ( b.m_n_particles > 0 );
        chunk_table.push_back( { write_offset , store_comp_size ? static_cast<uint32_t>(b.m_data.size()) : static_cast<uint32_t>(b.m_raw_size) , static_cast<int>(b.m_n_particles) } );
        file.increment_user_offset_bytes( write_offset - file.user_offset() );
        file.write_bytes( b.m_data.data() , b.m_data.size() );
        write_offset += b.m_data.size();
      }
    }
    
    return next_offset;
  }

  template<class LDBG, class GridT, class OptionalDumpFilter, class ... DumpFieldIds>
  static inline void write_dump_transcode( MPI_Comm comm, LDBG& ldbg, const GridT& grid, const Domain& domain, double phystime, long timestep, const std::string& filename
                                         , int compression_level, OptionalDumpFilter dump_filter, FieldSet< DumpFieldIds... > dump_fields , size_t max_part_size = MpiIO::DEFAULT_MAX_FILE_SIZE )
  {
    using RealDumpFields = FieldSetIntersection< typename GridT::Fields , FieldSet< DumpFieldIds... > >;
    using ParticleTuple = onika::soatl::FieldTupleFromFieldIds< RealDumpFields >;
    using StorageTuple = std::remove_cv_t< std::remove_reference_t< decltype( dump_filter.encode( ParticleTuple{} ) ) > >;
    using CompactFieldArrays = compact_field_arrays_from_tuple_t< StorageTuple >;
    
    //static constexpr RealDumpFields real_dump_fields = {};
    static constexpr size_t WRITE_BUFFER_SIZE = 65536; // number of particles per write

    std::filesystem::path filepath = filename;
    lout << "============ "<< filepath.stem() <<" ============" << std::endl ;

    // number of threads used for compression
    SimDumpHeader header = {};
    const unsigned int compression_threads = omp_get_max_threads();
    lout << "compr. level   = "<<compression_level <<std::endl
         << "compr. threads = "<< compression_threads <<std::endl;

    // initialize header
    StorageTuple{}.apply_fields( SimDumpInitializer{header} );
    header.m_tuple_size = sizeof(StorageTuple);
    header.m_domain = domain;
    header.m_time = phystime;
    header.m_time_step = timestep;
    size_t particle_bytes = 0;
    for(unsigned int i=0;i<header.m_nb_fields;i++)
    {
      //ldbg << "\tfield "<< header.m_fields[i] <<" size="<<header.m_field_size[i]<<std::endl;
      particle_bytes += header.m_field_size[i];
    }
    lout << "particle size  = "<< header.m_tuple_size<<std::endl
         << "part. payload  = "<< particle_bytes<<std::endl;

    // get MPi configuration
    int np=1, rank=0;
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&rank);

/*
    ldbg << "StorageTuple :\n";
    StorageTuple{}.apply_fields( print_tuple_field(ldbg) );
    ldbg << "ParticleTuple :\n";
    ParticleTuple{}.apply_fields( print_tuple_field(ldbg) );
*/

    // get grid setup
    IJK dims = grid.dimension();
    ssize_t gl = grid.ghost_layers();
    IJK gstart { gl, gl, gl };
    IJK gend = dims - IJK{ gl, gl, gl };
    IJK dims_no_ghost = dims - 2*gl;
    size_t n_cells_no_ghost = grid_cell_count( dims_no_ghost );
    auto cells = grid.cells();

    // count local number of particles
    unsigned long long number_of_particles = 0;
    GRID_BLOCK_FOR_BEGIN(dims,gstart,gend,i,_)
    {
      number_of_particles += cells[i].size();
    }
    GRID_BLOCK_FOR_END

    // compute particle index offsets for each MPI process' sub domain
    unsigned long long number_of_chunks = compression_threads + ( number_of_particles / WRITE_BUFFER_SIZE );
    unsigned long long all_number_of_chunks = 0;
    unsigned long long all_number_of_particles = 0;
    MPI_Allreduce(&number_of_particles,&all_number_of_particles,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    MPI_Allreduce(&number_of_chunks,&all_number_of_chunks,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    all_number_of_chunks += 1; // ensures there's enaough slots in chunk table, including a free slot @ end to store file size

    unsigned long long total_uncompressed_bytes = 0;
    unsigned long long total_compressed_bytes = 0;

    auto T0 = std::chrono::high_resolution_clock::now();

    // open output file
    // First, create directory
//    ldbg << filename << std::endl;
    lout << "file path      = "<<filename<<std::endl;
    if( rank == 0 )
    {
      if( filepath.has_parent_path() )
      {
        ldbg << "Create directory "<< filepath.parent_path() << std::endl;
        std::filesystem::create_directories( filepath.parent_path() );
      } 
    }
    MPI_Barrier(comm); // try to wait until other smp nodes see created directory (not guarented though)
    if( filepath.has_parent_path() && ! std::filesystem::is_directory( std::filesystem::status( filepath.parent_path() ) ) )
    {
      fatal_error() << "Directory " << filepath.parent_path() << " does not exist (despite prior creation)" << std::endl;
    }

    // Second, create output file
    MpiIO file;
    file.open( comm, filename , "w" , max_part_size );

    // write header and optional header
    std::vector<DataChunkItem> local_chunk_table;
    if( rank == 0 )
    {
      std::string optional_header;
      size_t optional_header_raw_size = 0;
      {
        std::ostringstream oss;
        dump_filter.write_optional_header( buffer_write_func(oss) );
        auto str = oss.str();
        optional_header_raw_size = str.size();
        MpiIO::compress_buffer( str.data(), optional_header_raw_size , optional_header, compression_level );
      }
      ldbg << "optional header size = "<< optional_header_raw_size << ", compressed = "<<optional_header.size() <<std::endl;

      assert( sizeof(DataChunkItem) == sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) ); // check for backward compatibility

      header.m_nb_particles = all_number_of_particles;
      header.m_optional_offset = sizeof( header );
      header.m_optional_header_size = optional_header_raw_size;
      header.m_table_offset = header.m_optional_offset + optional_header.size();
      header.m_chunk_count = all_number_of_chunks;
      size_t chunk_table_size = header.m_chunk_count * sizeof(DataChunkItem) ;
      header.m_data_offset = header.m_table_offset + chunk_table_size ;
      
      file.write( &header );
      file.increment_offset( &header );
      file.write_bytes( optional_header.data() , optional_header.size() );
      file.increment_user_offset_bytes( optional_header.size() );
      
      // skip chunk table, will write it afterward
      file.increment_user_offset_bytes( chunk_table_size ); 
      
      total_uncompressed_bytes += sizeof(header) + optional_header_raw_size + chunk_table_size ;
      total_compressed_bytes += header.m_data_offset;
    }
    MPI_Bcast( &header , sizeof(header) , MPI_CHAR , 0 , comm );    
    ldbg << "wrote header, number_of_particles="<< header.m_nb_particles<< ", number_of_chunks="<<header.m_chunk_count<<", data_offset="<< header.m_data_offset <<std::endl;
    if( rank != 0 )
    {
      file.increment_user_offset_bytes( header.m_data_offset );
    }
    
    // thread shared counters and buffers
    assert( file.user_offset() == static_cast<MPI_Offset>(header.m_data_offset) );
    unsigned long long global_write_offset = header.m_data_offset;
    unsigned long long global_next_offset = global_write_offset;
    std::atomic<size_t> n_zero_out = 0;
    unsigned long long particles_written = 0;
    std::atomic<int64_t> next_cell = 0;
    std::atomic<int64_t> n_end_of_grid = 0;
    std::vector< CompressionBuffer > comp_buffers(compression_threads);
    std::vector< CompressionBuffer > send_comp_buffers(compression_threads);

    // main write process
    dump_filter.initialize_write();
    
#   pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int nt = omp_get_num_threads();
      assert( tid>=0 && tid < int(comp_buffers.size()) );    

      // thread local buffers
      std::vector< StorageTuple > particle_buffer;
      particle_buffer.reserve(WRITE_BUFFER_SIZE + WRITE_BUFFER_SIZE/4 );
      CompactFieldArrays particle_buffer_array;
      std::vector<uint8_t> optional_data_buffer;    

      ssize_t write_buf_cur = 0;    
      
      bool end_of_grid = false;
      do
      {
        ssize_t cell_i = next_cell.fetch_add(1);
        bool prev_end_of_grid = end_of_grid;
        end_of_grid = ( cell_i >= ssize_t(n_cells_no_ghost) );
        
        // Step 1 : accumulate cell's particles to buffer
        if( ! end_of_grid )
        {
          size_t i = grid_ijk_to_index( dims , grid_index_to_ijk(dims_no_ghost,cell_i) + gl );
          size_t n_particles = cells[i].size();      
          for(size_t j=0;j<n_particles;j++)
          {
            ParticleTuple tp;
            cells[i].read_tuple( j , tp );
            auto stp = dump_filter.encode( tp );
            if( Vec3d{stp[field::rx],stp[field::ry],stp[field::rz]} == Vec3d{0,0,0} ) { ++ n_zero_out; }
            particle_buffer.push_back(stp); ++write_buf_cur; // particle_buffer[write_buf_cur++] = stp;
            //particle_buffer.write_tuple( write_buf_cur++ , stp );        
          }
          const size_t optional_data_size = dump_filter.optional_cell_data_size(i);
					// Add extra data storage, data are stored like in a Move buffer, i.e. header, info, data
          if( optional_data_size > 0 )
          {
						const unsigned int old_size = optional_data_buffer.size();
						optional_data_buffer.resize(old_size + optional_data_size); // resize first
						uint8_t* buff_ptr = optional_data_buffer.data() + old_size;
						dump_filter.write_optional_cell_data(buff_ptr, i);
          }
        }

        // Step 2 : when particle buffer is big enough, make it ready to be written
        assert( particle_buffer.size() == size_t(write_buf_cur) );
        if( write_buf_cur>0 && ( write_buf_cur >= ssize_t(WRITE_BUFFER_SIZE) || end_of_grid ) )
        {
          // temporary write buffer to concatenante particle array and optional data, if any
          std::vector<uint8_t> write_tmp;
          uint8_t* write_buffer_ptr = nullptr;
          
          // compute storage size of particle array without actually allocating it
          particle_buffer_array.unsafe_make_view( nullptr , write_buf_cur );
          const size_t particle_array_bytes = particle_buffer_array.storage_size();
          particle_buffer_array.unsafe_reset();

          bool extended_storage = ! optional_data_buffer.empty();

          // allocate write buffer
          if( ! extended_storage )
          {
            particle_buffer_array.resize( write_buf_cur );
            particle_buffer_array.shrink_to_fit();
            write_buffer_ptr = (uint8_t*) particle_buffer_array.storage_ptr();
          }
          else
          {
            write_tmp.clear();
            write_tmp.resize( particle_array_bytes + optional_data_buffer.size() );
            write_buffer_ptr = write_tmp.data();
            particle_buffer_array.unsafe_make_view( write_buffer_ptr, write_buf_cur );
          }
          
          // copy particle tuples to particle_buffer_array
          for(ssize_t k=0;k<write_buf_cur;k++) { particle_buffer_array.write_tuple( k , particle_buffer[k] ); }
          
          if( extended_storage )
          {
            particle_buffer_array.unsafe_reset();
            std::memcpy( write_buffer_ptr + particle_array_bytes , optional_data_buffer.data() , optional_data_buffer.size() );
          }

          bool buffer_ready_to_use = false;
          while( ! buffer_ready_to_use )
          {
#           pragma omp critical(xnb_dump_io_write_buffer)
            {
              buffer_ready_to_use = ( (comp_buffers[tid].m_raw_size==0) && (comp_buffers[tid].m_n_particles==0) && comp_buffers[tid].m_data.empty() ) ;
            }
          }

          CompressionBuffer cb;
          cb.m_raw_size = particle_array_bytes + optional_data_buffer.size();
          cb.m_n_particles = extended_storage ? -write_buf_cur : write_buf_cur; // negative particle count signals presence of optional data
          MpiIO::compress_buffer( (const char*) write_buffer_ptr , cb.m_raw_size , cb.m_data , compression_level );

#         pragma omp critical(xnb_dump_io_write_buffer)
          {
            comp_buffers[tid] = std::move( cb );
            //ldbg << "buffer ready : "<<write_buf_cur<<" particles, size = "<<comp_buffers[tid].m_raw_size<<" (opt. "<<optional_data_buffer.size()<<") / "<<comp_buffers[tid].m_data.size()<<std::endl;
          }

          write_buf_cur = 0;
          particle_buffer_array.clear();
          particle_buffer.clear();
          optional_data_buffer.clear();
        }

#       pragma omp master
        {
          size_t uncompressed_bytes = 0;
          size_t compressed_bytes = 0;
          size_t npart = 0;
          unsigned int ready_buffers = 0;
#         pragma omp critical(xnb_dump_io_write_buffer)
          {
            for(int t=0;t<nt;t++)
            {
              if( comp_buffers[t].m_raw_size > 0 ) ++ ready_buffers;
              uncompressed_bytes += comp_buffers[t].m_raw_size;
              compressed_bytes += comp_buffers[t].m_data.size();
              npart += std::abs( comp_buffers[t].m_n_particles );
              send_comp_buffers[t] = std::move( comp_buffers[t] );
              comp_buffers[t].m_data.clear();
              comp_buffers[t].m_raw_size = 0;
              comp_buffers[t].m_n_particles = 0;              
            }
            total_uncompressed_bytes += uncompressed_bytes;
            total_compressed_bytes += compressed_bytes;
            particles_written += npart;
          }

          if( ready_buffers > 0 )
          {
            //ldbg << "write "<<ready_buffers<<" buffers, particles="<< npart <<", uncompressed="<<uncompressed_bytes<<", compressed="<<compressed_bytes <<std::endl;
            file.collective_update_file_parts();
            global_next_offset = collective_write( file, comm, np, global_write_offset, local_chunk_table, send_comp_buffers );
            global_write_offset = global_next_offset;
          }            
        }

        if( !prev_end_of_grid && end_of_grid )
        {
          bool buffer_ready_to_use = false;
          while( ! buffer_ready_to_use )
          {
#           pragma omp critical(xnb_dump_io_write_buffer)
            {
              buffer_ready_to_use = ( (comp_buffers[tid].m_raw_size==0) && (comp_buffers[tid].m_n_particles==0) && comp_buffers[tid].m_data.empty() ) ;
            }
          }
          ++ n_end_of_grid;
        }
      }
      while( n_end_of_grid.load() < nt );

    } // end of parallel section

    ldbg << "particles written locally = "<< particles_written <<std::endl;

    // keep on participating until all others have finished
    int nb_process_active = 0;
    do
    {
      nb_process_active = file.collective_update_file_parts( false );
      global_write_offset = global_next_offset;
      global_next_offset = collective_write( file, comm, np, global_write_offset, local_chunk_table, {} );
      ldbg << "Wait for others : global_next_offset="<< global_next_offset << " , nb_active="<<nb_process_active<< std::endl;
    }
    while( nb_process_active>0 || global_next_offset > global_write_offset );
    
    unsigned long long all_chunk_count = local_chunk_table.size();
    MPI_Allreduce(MPI_IN_PLACE,&all_chunk_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    lout << "chunk count    = "<< all_chunk_count << " / " << header.m_chunk_count <<std::endl;
    if( all_chunk_count > header.m_chunk_count )
    {
      lerr << "Too many chunks ("<<all_chunk_count <<") , can't fit in chunk table ("<<header.m_chunk_count <<" entries)"<<std::endl;
      std::abort();
    }
    
    std::vector<DataChunkItem> all_chunk_table( all_chunk_count );
    {
      std::vector<int> counts(np,0);
      int local_chunk_count = local_chunk_table.size();
      MPI_Gather( &local_chunk_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm );
      for(int i=0;i<np;i++) { counts[i] *= sizeof(DataChunkItem); }
      std::vector<int> displs(np,0);
      for(int i=1;i<np;i++) displs[i] = displs[i-1] + counts[i-1];
      MPI_Gatherv( local_chunk_table.data(), local_chunk_table.size()*sizeof(DataChunkItem), MPI_CHAR, all_chunk_table.data(), counts.data(), displs.data(), MPI_CHAR, 0, comm );
    }

    // let's pretend we're active again, because dum_filter.finalize() may invoke collective write operations
    file.set_user_offset_bytes( global_write_offset );
    file.update_file_part();
    file.collective_update_file_parts();
    dump_filter.finalize_write();

    if( rank == 0 )
    {    
      std::sort( all_chunk_table.begin(), all_chunk_table.end(), [](const DataChunkItem& a, const DataChunkItem& b)->bool{ return a.m_global_offset < b.m_global_offset; } );
      ldbg << "file total size = "<< global_write_offset << std::endl;
      all_chunk_table.resize( header.m_chunk_count , DataChunkItem{global_write_offset,0,0} );
      
      size_t check_particle_count = 0;
      for(size_t i=0;i<all_chunk_table.size();i++)
      {
        //ldbg << "chunk table #"<<i<<" : offset="<<all_chunk_table[i].m_global_offset<<", size="<<all_chunk_table[i].m_data_size<<", particles="<<std::abs( all_chunk_table[i].m_n_particles )<<std::endl;
        check_particle_count += std::abs( all_chunk_table[i].m_n_particles );
      }
      ldbg << "check_particle_count = " << check_particle_count << std::endl;
    }

    // process 0 will rewind to chunk table start, while all other processes are inactive
    file.set_user_offset_bytes( header.m_table_offset );
    file.update_file_part();
    file.collective_update_file_parts( rank == 0 );

    if( rank == 0 )
    {    
      file.update_file_part();
      file.write( all_chunk_table.data() , all_chunk_table.size() );
    }
 
    while( file.collective_update_file_parts( false ) );

    double write_time_s = (std::chrono::high_resolution_clock::now()-T0).count()/1000000000.0;


    //ldbg << n_zero_out << " particles located at (0,0,0)\n";
    if( n_zero_out > 1 )
    {
      lerr << n_zero_out << " particles located at (0,0,0), output file will be corrupted\n"<<std::flush;
      std::abort();
    }

    MPI_Allreduce(MPI_IN_PLACE,&total_uncompressed_bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);    
    MPI_Allreduce(MPI_IN_PLACE,&total_compressed_bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);    
    MPI_Allreduce(MPI_IN_PLACE,&particles_written,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
    
    assert( total_compressed_bytes == global_write_offset );
    
    lout << "particles      = " << onika::large_integer_to_string(particles_written) << std::endl
         << "file parts     = " << file.number_of_file_parts() <<std::endl
         << "total size     = " << onika::large_integer_to_string(global_write_offset) <<std::endl
         << "uncompressed   = " << onika::large_integer_to_string(total_uncompressed_bytes) << std::endl
         << "compr. ratio   = " << 100-((total_compressed_bytes*100)/total_uncompressed_bytes) <<"%" << std::endl
         << "throughput     = " << static_cast<long>( (total_uncompressed_bytes/(1024*1024)) / write_time_s ) << " Mb/s" << std::endl
         << "===============================================" << std::endl << std::endl;

    file.close();
  }

  template<class LDBG, class GridT, class DumpFieldSet, class OptionalDumpFilter = NullDumpOptionalFilter >
  static inline void write_dump( MPI_Comm comm, LDBG& ldbg, const GridT& grid, const Domain& domain, double phystime, long timestep, const std::string& filename
                               , int compression_level, DumpFieldSet dump_fields, OptionalDumpFilter dump_filter = {} , size_t max_part_size = MpiIO::DEFAULT_MAX_FILE_SIZE )
  {
    write_dump_transcode( comm, ldbg, grid, domain, phystime, timestep, filename, compression_level, dump_filter, dump_fields , max_part_size );
  }

  template<typename GridT, class FS, class OptionalDumpFilter = NullDumpOptionalFilter > class SimDumpWriter;

  template<typename GridT, class OptionalDumpFilter, class ... DumpFieldIds >
  class SimDumpWriter< GridT , FieldSet<DumpFieldIds...> , OptionalDumpFilter > : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT       , grid     , INPUT );
    ADD_SLOT( Domain      , domain   , INPUT );
    ADD_SLOT( std::string , filename , INPUT );
    ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
    ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );
    ADD_SLOT( long        , compression_level , INPUT , 0 , DocString{"Zlib compression level"} );

  public:
    inline void execute () override final
    {
      write_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, *compression_level, FieldSet< DumpFieldIds... >{} , OptionalDumpFilter{} );
    }
  };

}


