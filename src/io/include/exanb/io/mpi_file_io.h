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

#include <mpi.h>
#include <string>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>
#include <utility>
#include <sstream>

#ifndef XNB_MPIIO_PREOPEN_FILE_PARTS
#define XNB_MPIIO_PREOPEN_FILE_PARTS 4
#endif

namespace exanb
{

  struct MpioIOText
  {
    static inline constexpr size_t MAX_SAMPLE_SIZE = 1024-2;
    const char* const m_header = nullptr;
    const size_t m_header_size = 0;
    const char* const m_sample_format = nullptr;
    const size_t m_sample_size = 0;
    MPI_File m_file = ( MPI_File ) nullptr;

    bool open(MPI_Comm comm, const std::string& filename);
    bool close();   
    bool write_sample_buf( size_t idx , const char* buf );

    template<class... ArgsT>
    inline bool write_sample(size_t idx , const ArgsT& ... args )
    {
      char buf[MAX_SAMPLE_SIZE+2];
      [[maybe_unused]] int bufsize = std::snprintf( buf, m_sample_size+1, m_sample_format, args ... );
      assert( size_t(bufsize) == m_sample_size );
      buf[m_sample_size-1] = '\n';
      buf[m_sample_size  ] = '\0';
      return write_sample_buf( idx , buf );
    }
  };

  class MpiIO
  {
  public:
    static constexpr size_t DEFAULT_MAX_FILE_SIZE = 128ull * 1024ull * 1024ull * 1024ull; // maximum 128Gb per file
    static constexpr size_t MAX_IO_OPERATION_SIZE = 512ull * 1024ull * 1024ull; // maximum 512Mb per I/O operation
    static constexpr size_t PREOPEN_FILE_PARTS = XNB_MPIIO_PREOPEN_FILE_PARTS;

    inline void open(const std::string& filename, std::string rw, size_t block_size = DEFAULT_MAX_FILE_SIZE ) { open( MPI_COMM_WORLD, filename, rw, block_size ); }
    void open(MPI_Comm comm, const std::string& filename, std::string rw, size_t block_size = DEFAULT_MAX_FILE_SIZE );
    void close();
    int collective_update_file_parts(bool active_process=true); // returns number of actively writing processes
    MPI_File file_part();

    static inline void handle_error(int errcode);
    inline MPI_Offset current_offset() const { return m_current_offset; }
    inline MPI_Offset user_offset() const { return m_user_offset; }
    
    //------------------------------------------------------------------------------------------------
    void increment_offset_bytes( ssize_t n );
    void increment_user_offset_bytes( ssize_t n );
    void set_user_offset_bytes( ssize_t raw_bytes );
    void update_file_part();
    template<typename MPIIOType> inline void increment_offset( MPIIOType* buffer , ssize_t n = 1)
    {
      increment_user_offset_bytes( n * sizeof(MPIIOType) );
    }

    //------------------------------------------------------------------------------------------------
    void part_read_bytes(size_t offset, void* buffer, size_t n);
    void read_bytes(void* buffer, size_t n);
    template<typename MPIIOType> inline void read(MPIIOType* buffer, size_t n = 1 ) { read_bytes( buffer, n*sizeof(MPIIOType) ); }

    //------------------------------------------------------------------------------------------------
    void part_write_bytes(size_t offset, const void* buffer, size_t n);
    void write_bytes(const void* buffer, size_t n);
    template<typename MPIIOType> inline void write(const MPIIOType* buffer, size_t n = 1) { write_bytes( buffer, n*sizeof(MPIIOType) ); }
    
    static size_t compression_buffer_size(size_t n);
    static void compress_buffer(const char* s, size_t n, std::string& output, int compLevel);
    static void uncompress_buffer(const char* input, size_t compressed_size, char* output, size_t uncompressed_size);

    inline size_t number_of_file_parts() const { return m_writen_file_parts; }

  private:
    //std::vector< std::map< std::pair<size_t,size_t> , std::pair<size_t,size_t> > > m_compressed_chunk_map;

    // ----------- data memebers -----------------------
    MPI_Comm m_comm = MPI_COMM_WORLD;
    MPI_Offset m_current_offset = 0;
    MPI_Offset m_user_offset = 0;

    MPI_File m_file_parts[PREOPEN_FILE_PARTS];
    ssize_t m_file_part_index = 0;
    ssize_t m_total_file_parts = 0; // total number of file parts opened
    ssize_t m_writen_file_parts = 0; // number of file parts really accessed
    
    std::string m_filename;
    size_t m_max_file_size = DEFAULT_MAX_FILE_SIZE;
    size_t m_file_index = 0;
    bool m_access_read_only = true;
    //bool m_compressed = false;
  };
 
  struct MpiIOWriteFunc
  {
    MpiIO& file;

    template<class T>
    inline size_t operator () ( const T& obj )
    {
      file.write( &obj, 1 );
      file.increment_offset( &obj, 1 );
      return sizeof(T);
    }
    template<class T, class A>
    inline size_t operator () ( const std::vector<T,A>& obj )
    {
      size_t n = obj.size();
      file.write( &n, 1 );
      file.increment_offset( &n, 1 );
      file.write( obj.data(), n );
      file.increment_offset( obj.data(), n );
      return sizeof(size_t) + sizeof(T) * n;
    }
  };
  inline MpiIOWriteFunc mpiio_write_func(MpiIO& file) { return {file}; }

  struct MpiIOReadFunc
  {
    MpiIO& file;
    template<class T>
    inline size_t operator () ( T& obj )
    {
      file.read( &obj, 1 );
      file.increment_offset( &obj, 1 );
      return sizeof(T);
    }
    template<class T, class A>
    inline size_t operator () ( std::vector<T,A>& obj )
    {
      obj.clear();
      size_t n = 0;
      file.read( &n, 1 );
      file.increment_offset( &n, 1 );
      obj.resize( n );
      file.read( obj.data(), n );
      file.increment_offset( obj.data(), n );
      return sizeof(size_t) + sizeof(T) * n;
    }
  };
  inline MpiIOReadFunc mpiio_read_func(MpiIO& file) { return {file}; }

  struct BufferReadFunc
  {
    std::istringstream& file;
    template<class T>
    inline size_t operator () ( T& obj )
    {
      file.read( reinterpret_cast<char*>(&obj), sizeof(T) );
      return sizeof(T);
    }
  };
  inline BufferReadFunc buffer_read_func(std::istringstream& file) { return { file }; } //std::istringstream(std::string(buf,len))

  struct StringBufferWriteFunc
  {
    std::ostringstream& file;
    template<class T>
    inline size_t operator () ( const T& obj )
    {
      file.write( reinterpret_cast<const char*>(&obj), sizeof(T) );
      return sizeof(T);
    }
  };
  inline StringBufferWriteFunc buffer_write_func(std::ostringstream& file) { return { file }; } //std::istringstream(std::string(buf,len))

  struct MpiIONullReadWriteFunc
  {
    MpiIO& m_file;
    bool m_inc_offset = true;
    template<class T>
    inline constexpr size_t operator () ( const T& obj ) const
    {
      if( m_inc_offset ) m_file.increment_offset( &obj, 1 );
      return sizeof(T);
    }
  };
  inline MpiIONullReadWriteFunc mpiio_null_rw_func(MpiIO& file, bool inc_offset = true ) { return {file,inc_offset}; }
  
}

