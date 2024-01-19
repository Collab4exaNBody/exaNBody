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
#include <iomanip>
#include <exanb/core/basic_types_def.h>

namespace exanb
{
  inline std::string vtk_make_spaces(int n)
  {
    const std::string spaces = "                                                     ";
    assert( size_t(n) < spaces.length() );
    return spaces.substr(0,n);
  }

# define vtk_space_offset_two ::exanb::vtk_make_spaces(2)
# define vtk_space_offset_four ::exanb::vtk_make_spaces(4)
# define vtk_space_offset_six ::exanb::vtk_make_spaces(6)
# define vtk_space_offset_eight ::exanb::vtk_make_spaces(8)
# define vtk_space_offset_ten ::exanb::vtk_make_spaces(10)

  template<class T> struct ParaViewTypeId
  {
    using comp_type = T;
    static constexpr int ncomp = 0;
    static inline const char* str() { return "Void"; }
  };
  
  template<> struct ParaViewTypeId<uint8_t>  { using comp_type=uint8_t;  static constexpr int ncomp=1; static inline const char* str() { return "UInt8"; } };
  template<> struct ParaViewTypeId< int8_t>  { using comp_type= int8_t;  static constexpr int ncomp=1; static inline const char* str() { return "Int8"; } };

  template<> struct ParaViewTypeId<uint16_t> { using comp_type=uint16_t; static constexpr int ncomp=1; static inline const char* str() { return "UInt16"; } };
  template<> struct ParaViewTypeId< int16_t> { using comp_type= int16_t; static constexpr int ncomp=1; static inline const char* str() { return "Int16"; } };

  template<> struct ParaViewTypeId<uint32_t> { using comp_type=uint32_t; static constexpr int ncomp=1; static inline const char* str() { return "UInt32"; } };
  template<> struct ParaViewTypeId< int32_t> { using comp_type= int32_t; static constexpr int ncomp=1; static inline const char* str() { return "Int32"; } };

  template<> struct ParaViewTypeId<uint64_t> { using comp_type=uint64_t; static constexpr int ncomp=1; static inline const char* str() { return "UInt64"; } };
  template<> struct ParaViewTypeId< int64_t> { using comp_type= int64_t; static constexpr int ncomp=1; static inline const char* str() { return "Int64"; } };

  template<> struct ParaViewTypeId< float>   { using comp_type= float;   static constexpr int ncomp=1; static inline const char* str() { return "Float32"; } };
  template<> struct ParaViewTypeId<double>   { using comp_type=double;   static constexpr int ncomp=1; static inline const char* str() { return "Float64"; } };

  template<> struct ParaViewTypeId<exanb::Vec3d>    { using comp_type=double;   static constexpr int ncomp=3; static inline const char* str() { return "Float64"; } };

  template<> struct ParaViewTypeId<exanb::Mat3d>    { using comp_type=double;   static constexpr int ncomp=9; static inline const char* str() { return "Float64"; } };
}

