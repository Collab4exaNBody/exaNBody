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

#include <exanb/core/basic_types.h>
#include <onika/print_utils.h>
#include <exanb/core/value_streamer.h>
#include <iostream>
#include <array>
#include <type_traits>

namespace exanb
{
  // pretty printing
 
  struct Mat3dStreamer
  {
    const Mat3d& m_mat_ref;
    const char m_mat_begin = '{';
    const char m_mat_end = '}';
    const char m_col_begin = '(';
    const char m_col_end = ')';
    const char m_col_sep = ',';
    const char m_value_sep = ',';
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const
    {
      out<<m_mat_begin<<m_col_begin<<m_mat_ref.m11<<m_value_sep<<m_mat_ref.m12<<m_value_sep<<m_mat_ref.m13<<m_col_end<<m_col_sep
                      <<m_col_begin<<m_mat_ref.m21<<m_value_sep<<m_mat_ref.m22<<m_value_sep<<m_mat_ref.m23<<m_col_end<<m_col_sep
                      <<m_col_begin<<m_mat_ref.m31<<m_value_sep<<m_mat_ref.m32<<m_value_sep<<m_mat_ref.m33<<m_col_end
         <<m_mat_end;
      return out;
    }
  };

  inline onika::FormattedObjectStreamer<Mat3dStreamer> format_mat3d(const Mat3d& m, char mb='{', char me='}', char cb='(', char ce='}', char cs=',', char vs=',')
  {
    return onika::FormattedObjectStreamer<Mat3dStreamer>{ {m,mb,me,cb,ce,cs,vs} };
  }

  template<class T> struct ValueStreamerHelper<T,Vec3d>
  {
    static inline ValueStreamer<T>& to_stream(ValueStreamer<T>& out, const Vec3d& v)
    {
      *(out.buf++) = static_cast<T>( v.x );
      *(out.buf++) = static_cast<T>( v.y );
      *(out.buf++) = static_cast<T>( v.z );
      return out;
    }

    static inline ValueStreamer<T>& from_stream(ValueStreamer<T>& in, Vec3d& v)
    {
      v.x = static_cast<double>( *(in.buf++) );
      v.y = static_cast<double>( *(in.buf++) );
      v.z = static_cast<double>( *(in.buf++) );
      return in;
    }
  };

  template<class T> struct ValueStreamerHelper<T,Mat3d>
  {
    static inline ValueStreamer<T>& to_stream (ValueStreamer<T>& out, const Mat3d& m)
    {
      *(out.buf++) = static_cast<T>( m.m11 );
      *(out.buf++) = static_cast<T>( m.m12 );
      *(out.buf++) = static_cast<T>( m.m13 );
      *(out.buf++) = static_cast<T>( m.m21 );
      *(out.buf++) = static_cast<T>( m.m22 );
      *(out.buf++) = static_cast<T>( m.m23 );
      *(out.buf++) = static_cast<T>( m.m31 );
      *(out.buf++) = static_cast<T>( m.m32 );
      *(out.buf++) = static_cast<T>( m.m33 );
      return out;
    }

    static inline ValueStreamer<T>& from_stream(ValueStreamer<T>& in, Mat3d& m)
    {
      m.m11 = static_cast<double>( *(in.buf++) );
      m.m12 = static_cast<double>( *(in.buf++) );
      m.m13 = static_cast<double>( *(in.buf++) );
      m.m21 = static_cast<double>( *(in.buf++) );
      m.m22 = static_cast<double>( *(in.buf++) );
      m.m23 = static_cast<double>( *(in.buf++) );
      m.m31 = static_cast<double>( *(in.buf++) );
      m.m32 = static_cast<double>( *(in.buf++) );
      m.m33 = static_cast<double>( *(in.buf++) );
      return in;
    }
  };

}

namespace std
{
  inline std::ostream& operator << (std::ostream& out, const exanb::Vec3d& v)
  {
    out<<v.x<<','<<v.y<<','<<v.z;
    return out;
  }

  inline std::ostream& operator << (std::ostream& out, const exanb::Mat3d& m)
  {
    return out << onika::FormattedObjectStreamer<exanb::Mat3dStreamer>{ {m} };
  }

  inline std::ostream& operator << (std::ostream& out, const exanb::IJK& v)
  {
    out<<v.i<<','<<v.j<<','<<v.k;
    return out;
  }

  inline std::ostream& operator << (std::ostream& out, const exanb::GridBlock& v)
  {
    out<<'('<<v.start<<")-("<<v.end<<')';
    return out;
  }

  inline std::ostream& operator << (std::ostream& out, const exanb::AABB& b)
  {
    out<<"("<<b.bmin<<")-("<<b.bmax<<")";
    return out;
  }
}

