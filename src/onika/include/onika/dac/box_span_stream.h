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

