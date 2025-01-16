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
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/print_utils.h>

#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>

namespace exanb
{  
  LogStreamWrapper lout  { 1 , []() -> std::ostream& {return std::cout;} };
  LogStreamWrapper lerr  { 2 , []() -> std::ostream& {return std::cerr;} };

  static LogStreamWrapper null_log_stream;

  // pre-enable debug output before logging is configured when compiled for Debug target
  // lower case prefix 'dbg:' helps developper see if a debug message happens before logging configuration
# ifndef NDEBUG
  LogStreamWrapper ldbg_raw  { 1 , []() -> std::ostream& {return std::cout;} , [](std::ostream& os) -> std::ostream& {return os << "dbg: ";}  };
# else
  LogStreamWrapper ldbg_raw;
# endif

  
  void LogStreamWrapper::open( const std::string& file_name )
  {
    //std::cout << "LogStreamWrapper::open('"<< file_name << "')" << std::endl;
    m_out = [s=std::make_shared<std::ofstream>(file_name)]() -> std::ostream& {return *s;};
    m_fd = -1;
  }

  void LogStreamWrapper::set_filters( const std::unordered_set<size_t>& filters )
  {
    m_filter_enable = ! filters.empty();
    m_filters = filters;
  }

  LogStreamWrapper& LogStreamWrapper::filter(size_t label)
  {
    if( m_filter_enable && m_filters.find(label)==m_filters.end() )
    {
      return null_log_stream;
    }
    else
    {
      return *this;
    }
  }

  bool LogStreamWrapper::is_a_tty() const
  {
    if( m_fd >= 0 )
    {
      bool terminal = isatty( m_fd );
      return terminal;
    }
    else
    {
      return false;
    }
  }

  void LogStreamWrapper::progress_bar( const std::string mesg, double ratio )
  {
    if( is_a_tty() )
    {
      (*this) << mesg << " [" ;
      int marks = static_cast<int>(ratio*100.0);
      int i=0;
      for(;i<marks;i++) (*this) << '=';
      for(;i<100;i++) (*this) << ' ';
      (*this) << "]\r" << std::flush;
      if( ratio >= 1.0 ) (*this) << std::endl;
    }
    else
    {
      if( ratio == 0.0 ) { (*this) << mesg << std::flush; }
      else if( ratio >= 1.0 ) { (*this) << " done" << std::endl; }
      else { (*this) << "." << std::flush; }
    }
  }


  void configure_logging(bool debug, bool parallel_log,
                         std::string out_file_name,
                         std::string err_file_name,
                         std::string dbg_file_name,
                         int rank, int mpi_size)
  {
    if( !out_file_name.empty() && mpi_size>1 )
    {
      out_file_name = format_string("%s.%03d",out_file_name,rank );
    }
    if( !err_file_name.empty() && mpi_size>1 )
    {
      err_file_name = format_string("%s.%03d",err_file_name,rank );
    }
    if( !dbg_file_name.empty() && mpi_size>1 )
    {
      dbg_file_name = format_string("%s.%03d",dbg_file_name,rank );
    }

    // ============== configure logging =============
    if( rank == 0 || parallel_log )
    {
      if( ! out_file_name.empty() ) { lout.open(out_file_name); }
      if( ! err_file_name.empty() ) { lerr.open(err_file_name); }
      if(parallel_log) 
      {
        lout.m_begin_line = [rank](std::ostream& out) -> std::ostream& { return out<<exanb::format_string("P%03d: ",rank); } ;
        lerr.m_begin_line = [rank](std::ostream& out) -> std::ostream& { return out<<exanb::format_string("P%03d: ERR: ",rank); } ;
      }
      else
      {
        lerr.m_begin_line = [](std::ostream& out) -> std::ostream& { return out<<"ERR: "; } ;
      }
    }
    else
    {
      lout.m_out = nullptr;
      lerr.m_out = nullptr;
    }

    bool disable_dbg_output = false;
    if( debug )
    {
      if( rank == 0 || parallel_log )
      {
        if( ! dbg_file_name.empty() ) { ldbg_raw.open(dbg_file_name); }
        else { ldbg_raw.m_out = []() -> std::ostream& {return std::cout;} ; }
      }
      if( parallel_log )
      {
        ldbg_raw.m_begin_line = [rank](std::ostream& out) -> std::ostream& { return out<<exanb::format_string("P%03d: DBG: ",rank); } ;
      }
      else if( rank == 0 )
      {
        ldbg_raw.m_begin_line = [](std::ostream& out) -> std::ostream& { return out<<"DBG: "; } ;
      }
      else { disable_dbg_output = true; }
    }
    else { disable_dbg_output = true; }
    
    if( disable_dbg_output )
    {
      ldbg_raw.m_out = nullptr;
      ldbg_raw.m_begin_line = nullptr;
    }
    
    // default format
    lout << default_stream_format;
    ldbg_raw << default_stream_format;
  }

  // create a log filter for debug messages out of operator scope
  LogStreamFilterHelper ldbg { ::exanb::ldbg_raw , std::numeric_limits<uint64_t>::max() };

  FatalErrorLogStream::~FatalErrorLogStream()
  {
    using namespace std::chrono_literals;
    lerr_stream()
      << "*****************************************" << std::endl
      << "************* FATAL ERROR ***************" << std::endl
      << "*****************************************" << std::endl
      << "*** " << m_oss.str() << std::flush
      << "*****************************************" << std::endl;
    std::this_thread::sleep_for(500ms);
    std::abort();
  }


}

