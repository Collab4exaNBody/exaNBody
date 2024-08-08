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
#include <exanb/core/print_utils.h>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <mutex>

namespace exanb
{  
  std::ostream& default_stream_format(std::ostream& out)
  {
    out << std::scientific << std::setprecision(10) << std::boolalpha;
    return out;
  }

  
  namespace details
  {
    static FileAppendWriteBuffer g_FileAppendWriteBuffer_instance{};
  }

  void FileAppendWriteBuffer::append_to_file(const std::string& filename, const std::string& buffer, const bool forceappend)
  {
#   pragma omp critical(FileAppendWriteBuffer_append_to_file)
    {
      const bool file_exists = std::ifstream(filename).good();
      const bool create_file = (!file_exists) || (!forceappend);
      if( (m_create.find(filename)==m_create.end()) && create_file ) { m_create[filename]=true; }
      m_write_buffer[filename] += buffer;
      if( m_write_buffer[filename].size() >= s_max_buffer_size )
      {
        flush();
      }
    }
  }
  
  void FileAppendWriteBuffer::flush()
  {
    static std::mutex s_flush_mutex;
    s_flush_mutex.lock();
    flush_singlethread();
    s_flush_mutex.unlock();
  }

  void FileAppendWriteBuffer::flush_singlethread()
  {
    for(auto& p : m_write_buffer)
    {
      if( m_create[p.first] || ! p.second.empty() )
      {
	      std::ofstream out;
	      if( m_create[p.first] ) { out.open(p.first.c_str()); m_create[p.first]=false; }
	      else { out.open(p.first.c_str(),std::fstream::app); }
	      out << p.second;
	      p.second.clear();
      }
    }
  }

  FileAppendWriteBuffer& FileAppendWriteBuffer::instance()
  {
    return details::g_FileAppendWriteBuffer_instance;
  }

  FileAppendWriteBuffer::~FileAppendWriteBuffer()
  {
    flush_singlethread();
  }

}

