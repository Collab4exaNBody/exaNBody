#pragma once

#include <cstring>
#include <utility>

namespace onika
{
  namespace task
  {

    template<class StreamT , int Nin, int ... Is>
    static inline StreamT& print_depend_indices( StreamT& out , std::integer_sequence<int,Nin,Is...> )
    {
      int c=0;
      out<<"in:";
      ( ... , ( ( (c++) < Nin ) ? (out << " " << Is) : out ) );
      out<<" , inout:";
      c=0;
      ( ... , ( ( (c++) >= Nin ) ? (out << " " << Is) : out ) );
      return out;
    }

    inline const char* tag_filter_out_path(const char* tag)
    {
      static const char * no_tag = "<no-tag>";
      if( tag == nullptr ) return no_tag;
      const char* s = std::strrchr( tag , '/' );
      if( s == nullptr ) return tag;
      return s+1;
    }
     
  }
}
