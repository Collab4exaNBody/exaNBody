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
#include <exanb/io/mpi_file_io.h>
#include <onika/log.h>

#include <iostream>
#include <sstream>
#include <cassert>
#include <vector>
#include <fstream>
#include <cstring>
#include <limits>

#include <zlib.h>//lib for compression

namespace exanb
{


  // *********** Simple parallel ASCII output file writer ****************
  
  bool MpioIOText::open(MPI_Comm comm, const std::string& filename)
  {
    MPI_File_open(comm,filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR , MPI_INFO_NULL, & m_file);
    int rank=0;
    MPI_Comm_rank(comm,&rank);
    if( rank == 0 )
    {
      MPI_Status status;
      MPI_File_write_at( m_file, 0, m_header, m_header_size, MPI_CHAR, &status);
      int count = 0;
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
      if( size_t(count) != m_header_size ) return false;
    }
    return true;
  }
  
  bool MpioIOText::close()
  {
    MPI_File_close( & m_file );
    return true;
  }
  
  bool MpioIOText::write_sample_buf(size_t idx , const char* buf )
  {
    MPI_Status status;
    int count = 0;
    MPI_File_write_at( m_file , m_header_size + ( idx * m_sample_size ) , buf , m_sample_size , MPI_CHAR , &status );
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
    return size_t(count) == m_sample_size ;
  }



  // *********** Parallel dump file writer with multithreaded compression ****************
  
  void MpiIO::handle_error(int errcode)
  {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen = 0;
    MPI_Error_string(errcode, msg, &resultlen);
    lerr << " Error in MpiIO : " << msg << std::endl;
    std::abort();
  }

  void MpiIO::close()
  {
    m_current_offset = 0; 
    for(size_t i=0;i<PREOPEN_FILE_PARTS;i++)
    {
      MPI_File_close( & m_file_parts[i] );
    }
    if( ! m_access_read_only )
    {
      MPI_Allreduce(MPI_IN_PLACE,&m_total_file_parts,1,MPI_LONG_LONG,MPI_MAX,m_comm);
      MPI_Allreduce(MPI_IN_PLACE,&m_writen_file_parts,1,MPI_LONG_LONG,MPI_MAX,m_comm);
      int rank=0;
      MPI_Comm_rank(m_comm,&rank);
      if( rank == 0 )
      {
        for(unsigned int fi=m_writen_file_parts;fi<m_total_file_parts;fi++)
        {
          std::ostringstream oss;
          oss << m_filename;
          if( fi != 0 ) oss<<"."<<fi;
          auto fname = oss.str();
          ldbg << "DELETE file part "<<fname<<std::endl;
          MPI_File_delete( fname.c_str(), MPI_INFO_NULL );
        }
      }
    }
  }

  void MpiIO::open(MPI_Comm comm, const std::string& filename, std::string rw, size_t block_size)
  {
    m_comm = comm;
  
    m_max_file_size = block_size;
    m_filename = filename;
    m_access_read_only = ( (rw=="r") || (rw=="R") );
    
    m_file_index = 0;
    m_file_part_index = 0;

    size_t std_file_size = 0;
    if( m_access_read_only )
    {
      std::ifstream fin( m_filename );
      fin.seekg( 0 , std::ios_base::end );
      std_file_size = fin.tellg();
      fin.close();
    }

    int rc = MPI_File_open(m_comm,filename.c_str(), m_access_read_only ? MPI_MODE_RDONLY : ( MPI_MODE_CREATE | MPI_MODE_RDWR ) , MPI_INFO_NULL, & m_file_parts[0]);
    if( rc != MPI_SUCCESS )
    {
      lerr << "MpiIO: unable to open file '" << filename << "'" << std::endl << std::flush ;
      std::abort();
    }
    
    if( m_access_read_only )
    {
      MPI_File_seek( m_file_parts[0] , 0 , MPI_SEEK_END );
      MPI_Offset file_size = 0;
      MPI_File_get_position( m_file_parts[0], &file_size );
      m_max_file_size = file_size;
      MPI_File_seek( m_file_parts[0] , 0 , MPI_SEEK_SET );
      if( m_max_file_size != std_file_size )
      {
        lerr<<"Warning : MPI_IO detects file size "<<m_max_file_size<<", but std::ifstream tells "<<std_file_size<<", using later one"<< std::endl;
        m_max_file_size = std_file_size;
      }
      //std::cout<<"read file : file part size = "<<m_max_file_size<<std::endl;
    }
    m_current_offset = 0;
    
    // pre-open next file parts to avoid deadlocks
    for(size_t i=1;i<PREOPEN_FILE_PARTS;i++)
    {
      std::ostringstream oss;
      size_t fi = m_file_part_index + i;
      oss << m_filename;
      if( fi != 0 ) oss<<"."<<fi;
      auto fname = oss.str();
      int rc = MPI_File_open( m_comm, fname.c_str(), m_access_read_only ? MPI_MODE_RDONLY : ( MPI_MODE_CREATE | MPI_MODE_RDWR ) , MPI_INFO_NULL, & m_file_parts[i] );
      if( rc != MPI_SUCCESS )
      {
        if( !m_access_read_only )
        {
          lerr << "MpiIO: unable to open file part '" << fname << "'" << std::endl << std::flush ;
          std::abort();
        }
        else
        {
          m_file_parts[i] = ( MPI_File ) nullptr;
        }
      }
      else
      {
        if( ssize_t(fi) >= m_total_file_parts ) m_total_file_parts = fi+1;
      }
    }
  }

  void MpiIO::increment_offset_bytes( ssize_t n )
  {
    m_current_offset += n;
  }

  void MpiIO::increment_user_offset_bytes( ssize_t raw_bytes )
  {
    m_user_offset += raw_bytes;
    increment_offset_bytes( ssize_t(m_user_offset) - ssize_t(m_current_offset) );
  }

  void MpiIO::set_user_offset_bytes( ssize_t raw_bytes )
  {
    m_user_offset = raw_bytes;
    increment_offset_bytes( ssize_t(m_user_offset) - ssize_t(m_current_offset) );
  }

  int MpiIO::collective_update_file_parts( bool active_process )
  {
    long long desired_frame_min = m_file_index;
    long long desired_frame_max = m_file_index;
    
    if( ! active_process )
    {
      desired_frame_min = std::numeric_limits<long long>::max();
      desired_frame_max = -1;
    }
    
    MPI_Allreduce(MPI_IN_PLACE,&desired_frame_min,1,MPI_LONG_LONG,MPI_MIN,m_comm);
    MPI_Allreduce(MPI_IN_PLACE,&desired_frame_max,1,MPI_LONG_LONG,MPI_MAX,m_comm);
    
    if( desired_frame_min > desired_frame_max )
    {
      ldbg <<"No process required any file part" <<std::endl<<std::flush;
      return 0;
    }
    
    if( (desired_frame_max - desired_frame_min) >= ssize_t(PREOPEN_FILE_PARTS) )
    {
      fatal_error() << "Processes require file part range ["<<desired_frame_min<<";"<<desired_frame_max<<"] which impossible to satisify"<<std::endl;
    }

    long long all_start_index = desired_frame_min;
    if( all_start_index < 0 )
    {
      fatal_error() << "invalid file part start index "<<all_start_index<<std::endl;
    }

    if( all_start_index != m_file_part_index )
    {
      ldbg <<"shift pre-open file parts : "<<m_file_part_index<<" -> "<< all_start_index <<std::endl<<std::flush;
      for(size_t i=0;i<PREOPEN_FILE_PARTS;i++)
      {
        MPI_File_close( & m_file_parts[i] );
      }
      m_file_part_index = all_start_index;
      for(size_t i=0;i<PREOPEN_FILE_PARTS;i++)
      {
        std::ostringstream oss;
        size_t fi = m_file_part_index + i;
        oss << m_filename;
        if( fi != 0 ) oss<<"."<<fi;
        auto fname = oss.str();
        int rc = MPI_File_open( m_comm, fname.c_str(), m_access_read_only ? MPI_MODE_RDONLY : ( MPI_MODE_CREATE | MPI_MODE_RDWR ) , MPI_INFO_NULL, & m_file_parts[i] );
        if( rc != MPI_SUCCESS )
        {
          if( !m_access_read_only )
          {
            lerr << "MpiIO: unable to open file part '" << fname << "'" << std::endl << std::flush ;
            std::abort();
          }
          else
          {
            m_file_parts[i] = ( MPI_File ) nullptr;
          }
        }
        else
        {
          if( ssize_t(fi) >= m_total_file_parts ) m_total_file_parts = fi+1;
        }
      }
    }
    
    int nb_active = active_process;
    MPI_Allreduce(MPI_IN_PLACE,&nb_active,1,MPI_INT,MPI_SUM,m_comm);
    return nb_active;
  }

  MPI_File MpiIO::file_part()
  {
    if( ssize_t(m_file_index) < m_file_part_index || m_file_index >= m_file_part_index+PREOPEN_FILE_PARTS )
    {
      fatal_error() << "access to a file part that has not been pre-opened : file_index=" <<m_file_index<<" , m_file_part_index="<<m_file_part_index<< std::endl;
    }
    return m_file_parts[ m_file_index - m_file_part_index ];
  }

  void MpiIO::update_file_part()
  {
    m_file_index = m_current_offset / m_max_file_size;
    if( ssize_t(m_file_index) >= m_writen_file_parts ) m_writen_file_parts = m_file_index+1;
  }

  void MpiIO::part_read_bytes( size_t offset, void* buffer, size_t n )
  {
    if( n > MAX_IO_OPERATION_SIZE )
    {
      lerr<<"MpiIO read error : exceeded maximum bytes per I/O operation : "<<n<<" > "<<MAX_IO_OPERATION_SIZE<<std::endl;
      std::abort();
    }
    if( (offset+n) > m_max_file_size )
    {
      lerr<<"MpiIO read error : read offset beyond file part: "<< (offset+n) <<" > "<<m_max_file_size <<std::endl;
      std::abort();
    }
    if( n>0 && file_part() == nullptr )
    {
      lerr<<"MpiIO read error : attempt to read missing file part #"<< m_file_index <<std::endl;
      std::abort();
    }
    
    MPI_Status status;
    MPI_File_read_at( file_part(), offset , buffer, n, MPI_UNSIGNED_CHAR, &status );
    int count = 0;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
    if( size_t(count) != n )
    {
      lerr<<"MpiIO read error : expected "<<n<<" bytes, got "<<count<<"\n";
      std::abort();
    }
  }

  void MpiIO::read_bytes(void* _buffer, size_t n)
  {
    uint8_t* buffer = (uint8_t*) _buffer; // for correct pointer arithmetic
    while( n > ( m_max_file_size - (m_current_offset % m_max_file_size) ) )
    {
      size_t to_read = m_max_file_size - ( m_current_offset % m_max_file_size );
      assert( to_read > 0 && to_read <= n );
      read_bytes( buffer , to_read );
      increment_offset_bytes( to_read );
      buffer += to_read;
      n -= to_read;
    }
    if(n==0) return;
  
    update_file_part();
    assert( m_current_offset/m_max_file_size == m_file_index );
    assert( ( m_current_offset - m_file_index*m_max_file_size ) == ( m_current_offset % m_max_file_size ) );

    size_t total_to_read = n;
    size_t read_offset = 0;
    while( total_to_read > 0 )
    {
      size_t to_read = total_to_read;
      if( to_read > MAX_IO_OPERATION_SIZE ) to_read = MAX_IO_OPERATION_SIZE;
      part_read_bytes( ( m_current_offset % m_max_file_size ) + read_offset , buffer, to_read );
      buffer = buffer + to_read;
      total_to_read = total_to_read - to_read;
      read_offset = read_offset + to_read;
    }
      
  }

  void MpiIO::part_write_bytes(size_t offset, const void* buffer, size_t n)
  {
    if( n > MAX_IO_OPERATION_SIZE )
    {
      lerr<<"MpiIO write error : exceeded maximum bytes per I/O operation : "<<n<<" > "<<MAX_IO_OPERATION_SIZE<<std::endl;
      std::abort();
    }
    if( (offset+n) > m_max_file_size )
    {
      lerr<<"MpiIO write error : write offset beyond file part: "<< (offset+n) <<" > "<<m_max_file_size <<std::endl;
      std::abort();
    }
    //ldbg << "part_write_bytes( "<<offset<<" , @"<<buffer<<" , "<<n<<" )"<<std::endl;
    MPI_Status status;
    MPI_File_write_at( file_part(), offset, buffer, n, MPI_UNSIGNED_CHAR, &status );
    int count = 0;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
    if( size_t(count) != n )
    {
      lerr<<"MpiIO write error\n";
      std::abort();
    }

  }

  void MpiIO::write_bytes(const void* _buffer, size_t n)
  {
    uint8_t* buffer = (uint8_t*) _buffer; // for correct pointer arithmetic
    while( n > ( m_max_file_size - (m_current_offset % m_max_file_size) ) )
    {
      size_t to_write = m_max_file_size - ( m_current_offset % m_max_file_size );
      assert( to_write > 0 && to_write <= n );
      //std::cout<<"sub write "<<to_write<<" / "<<n<<" @"<<(void*)buffer<<std::endl;
      write_bytes( buffer , to_write );
      increment_offset_bytes( to_write );
      buffer += to_write;
      n -= to_write;
    }
    if(n==0) return;

    update_file_part();
    assert( m_current_offset/m_max_file_size == m_file_index );
    assert( ( m_current_offset - m_file_index*m_max_file_size ) == ( m_current_offset % m_max_file_size ) );
    
    part_write_bytes( m_current_offset % m_max_file_size , buffer, n );
  }

  size_t MpiIO::compression_buffer_size( size_t n )
  {
    return static_cast<size_t>( (n * 1.001) + 12 );
  }

  void MpiIO::compress_buffer(const char* s, size_t n, std::string& compression_buffer, int comp)
  {
    if( comp == 0 || n == 0 )
    {
      compression_buffer.assign( s , s+n );
      return;
    }

    compression_buffer.resize( compression_buffer_size(n) );
    unsigned long cs = compression_buffer.size();
          
    bool compression_ok = compress2( (Bytef*) compression_buffer.data() , &cs , (const Bytef*) s , n, comp ) == Z_OK;

    if( ! compression_ok )
    {
      lerr<<"MpiIO error : compression failed" <<std::endl;
      std::abort();
    }
    // std::cout<<"compress buffer: "<<n<<" -> "<<cs<<std::endl;
    if( cs >= n ) // failed to compress, output raw data
    {
      cs = n;
      compression_buffer.resize(n);
      std::memcpy( compression_buffer.data(), s, n );
    }
    else
    {
      compression_buffer.resize( cs );
    }
  }

  void MpiIO::uncompress_buffer(const char* input, size_t compressed_size, char* output, size_t uncompressed_size)
  {
    unsigned long buffer_size = uncompressed_size;
    int rc = uncompress( (Bytef*) output , &buffer_size , (const Bytef*) input , compressed_size );
    assert( uncompressed_size == buffer_size );

    if( rc != Z_OK )
    {
      lerr<<"MpiIO error : decompression failed" <<std::endl;
      std::abort();
    }
    // std::cout<<"uncompress buffer: "<<compressed_size<<" -> "<<buffer_size<<std::endl;
  }

}

