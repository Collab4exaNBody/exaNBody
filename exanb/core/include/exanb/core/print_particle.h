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

#include <onika/soatl/field_tuple.h>
#include <onika/print_utils.h>

#include <cstdint>
#include <iomanip>

namespace exanb
{

  // =================== utility functions ==========================
  namespace details
  {
    template<typename T> static inline T print_convert(const T& x) { return x; }
    static inline int print_convert(const int8_t& x) { return x; }
    static inline int print_convert(const uint8_t& x) { return x; }

    template<typename T>
    static inline std::string convert_value_to_string( const T& x )
    {
      std::ostringstream oss;
      oss<< onika::default_stream_format;
      onika::print_if_possible( oss , print_convert(x) , "???" );
      return oss.str();
    }

  }

  template<typename StreamT, typename... field_id>
  static inline void print_particle(StreamT& out, const onika::soatl::FieldTuple<field_id...> & particle, bool brief=true)
  {
    if( brief )
    {
      int count=0;
      (...,(
        out << ( ( (count++)>0 ) ? " " : "" )
            << onika::soatl::FieldId<field_id>::short_name()
            << "=" << details::convert_value_to_string(particle[onika::soatl::FieldId<field_id>()])
      ));
      out << std::endl;
    }
    else
    {
      (...,(
        out << onika::soatl::FieldId<field_id>::name()
            << " = " << details::convert_value_to_string(particle[onika::soatl::FieldId<field_id>()])
            << std::endl
      ));
    }
  }

}

