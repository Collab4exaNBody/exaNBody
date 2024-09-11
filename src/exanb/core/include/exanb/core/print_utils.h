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
#include <array>
#include <type_traits>
#include <cassert>
#include <unordered_map>

#include <exanb/core/type_utils.h>

namespace exanb
{  
  // utility to print a value to a stream if << operator is available or an alternate message otherwise+
  namespace stream_utils_details
  {
  
    template <class T, 
              class... X,                                 // useless, just for overload priority
              std::enable_if_t<sizeof...(X)==0,int> = 0 > // enable if no extra parameters
    static inline std::ostream& print_if_possible(std::ostream& os, const T&, const std::string& alt, X...)
    {
      return os << alt ;
    }

    template <class T,
              class = decltype( std::declval<std::ostream>() << std::declval<T>() ) > // enable if operator << is overloaded with T
    static inline std::ostream& print_if_possible(std::ostream& os, const T& value, const std::string&)
    {
      return os << value ;
    }

  }

  template<typename T>
  inline std::ostream& print_if_possible( std::ostream& os, const T& value, const std::string& alt_txt)
  {
    return stream_utils_details::print_if_possible(os,value,alt_txt);
  }


  template<class StreamT>
  inline StreamT& spaces( StreamT& out , size_t n, char c=' ' )
  {
    for(size_t i=0;i<n;i++) { out << c ; }
    return out;
  }


  // default formatting options
  std::ostream& default_stream_format(std::ostream& out);


  //**************************************************************************
  //*********** file write buffer to avoid excessive open/close **************
  //**************************************************************************
  class FileAppendWriteBuffer
  {
  public:
    void append_to_file(const std::string& filename, const std::string& buffer, bool forceappend = false);
    void flush();
    void flush_singlethread();
    ~FileAppendWriteBuffer();
    
    static FileAppendWriteBuffer& instance();
    
  private:
    static constexpr size_t s_max_buffer_size = 4096;
    
    std::unordered_map< std::string , bool > m_create;
    std::unordered_map< std::string , std::string > m_write_buffer;
  };

  //**************************************************************************
  //**************************************************************************
  //**************************************************************************

  template<class T>
  struct FormattedObjectStreamer
  {
    const T m_streamer;
  };
}

namespace std
{
  template<class T>
  inline std::ostream& operator << (std::ostream& out , const exanb::FormattedObjectStreamer<T>& st )
  {
    return st.m_streamer.to_stream(out);
  }
}

