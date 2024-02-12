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

namespace exanb
{  
  //**************************************************************************
  //*********** stream scalar values to/from temporary buffer ****************
  //**************************************************************************
  
  // very basic helper class to stream values into (or from) a buffer
  template<class T>
  struct ValueStreamer
  {
    T* buf = nullptr;
    const T* end = nullptr;
    template<size_t N> inline ValueStreamer( T (&array)[N] ) : buf(array) , end(array+N) {}
    template<size_t N> inline ValueStreamer(std::array<T,N>& array) : buf(array.data()) , end(array.data()+N) {}
  };

  template<class T, class U>
  struct ValueStreamerHelper
  {
    static inline ValueStreamer<T>& to_stream(ValueStreamer<T>& out, const U& u)
    {
      assert( out.buf != out.end );
      *(out.buf++) = static_cast<T>( u );
      return out;
    }
    static inline ValueStreamer<T>& from_stream(ValueStreamer<T>& in, U& u)
    {
      assert( in.buf != in.end );
      u = static_cast<U>( *(in.buf++) );
      return in;
    }
  };


  template<class T, class U> ValueStreamer<T>& operator << (ValueStreamer<T>& out, const U& u)
  {
    return ValueStreamerHelper<T,U>::to_stream( out , u );
  }
  template<class T, class U> ValueStreamer<T> operator << (const ValueStreamer<T>& out, const U& u)
  {
    ValueStreamer<T> tmp_streamer = out;
    return ValueStreamerHelper<T,U>::to_stream( tmp_streamer , u );
  }

  template<class T, class U> ValueStreamer<T>& operator >> (ValueStreamer<T>& in, U& u)
  {
    return ValueStreamerHelper<T,U>::from_stream( in , u );
  }
  template<class T, class U> ValueStreamer<T> operator >> (const ValueStreamer<T>& in, U& u)
  {
    ValueStreamer<T> tmp_streamer = in;
    return ValueStreamerHelper<T,U>::from_stream( tmp_streamer , u );
  }
  //**************************************************************************
  //**************************************************************************
  //**************************************************************************

}

