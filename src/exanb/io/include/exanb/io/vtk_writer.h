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

