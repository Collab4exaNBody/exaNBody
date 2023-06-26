#pragma once

#include <cstdint>
#include <onika/dac/box_span.h>
#include <onika/oarray_stream.h>
#include <onika/stream_utils.h>


// =========== data access control constants ===========
namespace onika
{

  template<size_t N,size_t G> struct PrintableFormattedObject< dac::box_span_t<N,G> >
  {
    const dac::box_span_t<N,G>& m_obj;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const
    {
      return out << "{low="<<format_array(m_obj.lower_bound)<<";size="<<format_array(m_obj.box_size)<<";border="<<m_obj.border<<"}";
    }
  };
  
  template<size_t N, size_t G>
  inline PrintableFormattedObject< dac::box_span_t<N,G> > format_box_span(const dac::box_span_t<N,G>& sp) { return { sp }; }

}

