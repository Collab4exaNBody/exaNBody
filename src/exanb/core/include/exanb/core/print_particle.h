#pragma once

#include <onika/soatl/field_tuple.h>
#include <exanb/core/print_utils.h>

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
      oss<< default_stream_format;
      print_if_possible( oss , print_convert(x) , "???" );
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

