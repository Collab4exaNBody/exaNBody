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

#include <onika/soatl/field_id.h>
#include <onika/math/basic_types_def.h>
#include <cstdint>
#include <cassert>
#include <yaml-cpp/yaml.h>

#ifndef XNB_PARTICLE_TYPE_INT
#define XNB_PARTICLE_TYPE_INT uint8_t
#endif

namespace exanb
{
  template<typename... field_ids> using FieldSet = onika::soatl::FieldIds< field_ids... > ;
  template<typename... field_sets> struct FieldSets {};

  using ParticleTypeInt = XNB_PARTICLE_TYPE_INT;
}

#define XNB_DECLARE_FIELD(__type,__name,__desc) \
namespace exanb { namespace field { struct _##__name {}; } } \
namespace onika { namespace soatl { template<> struct FieldId<::exanb::field::_##__name> { \
    using value_type = __type; \
    using Id = ::exanb::field::_##__name; \
    static inline const char* short_name() { return #__name ; } \
    static inline const char* name() { return __desc ; } \
    static inline void set_name(const std::string& s) { assert( s == short_name() ); };\
  }; } } \
namespace exanb { namespace field { static constexpr ::onika::soatl::FieldId<_##__name> __name; } }

#define XNB_DECLARE_ALIAS(A,F) namespace exanb { namespace field { using _##A=_##F; static constexpr ::onika::soatl::FieldId<_##A> A; } }

#define XNB_DECLARE_DYNAMIC_FIELD(__type,__name,__desc) \
namespace exanb { namespace field { template<size_t _GENERIC_IDX> struct _##__name { static inline constexpr auto GENERIC_IDX=_GENERIC_IDX; }; } } \
namespace onika { namespace soatl { template<size_t _GENERIC_IDX> struct FieldId<::exanb::field::_##__name<_GENERIC_IDX> > { \
    using value_type = __type; \
    using Id = ::exanb::field::_##__name<_GENERIC_IDX>; \
    static inline constexpr size_t NAME_MAX_LEN = 16; \
    char m_name[NAME_MAX_LEN] = #__name ; \
    inline void set_name(const std::string& s) { strncpy(m_name,s.c_str(),NAME_MAX_LEN); m_name[NAME_MAX_LEN-1]='\0'; };\
    inline const char* short_name() const { return m_name; } \
    inline const char* name() const { return m_name; } \
  }; } } \
namespace exanb { namespace field { template<size_t I> using __name##_nth=::onika::soatl::FieldId<_##__name<I> >; using __name = __name##_nth<0> ; inline auto mk_##__name (const std::string& s) { __name f; f.set_name(s); return f; } } }

// for dynamic fields, allows field name to be read from YAML conversion
namespace YAML
{
  template<class fid> struct convert< onika::soatl::FieldId<fid> >
  {
    static inline bool decode(const Node& node, onika::soatl::FieldId<fid>& v )
    {
      if( ! node.IsScalar() ) return false;
      std::string s = node.as<std::string>();
      v.set_name( s );
      return true;
    }
  };  
}

// default particle fields that are defined in namespace 'field'
// for rx field descriptor instance, use field::rx, for its type, use field::_rx
XNB_DECLARE_FIELD(uint64_t                 ,id   ,"particle id")
XNB_DECLARE_FIELD(::exanb::ParticleTypeInt ,type ,"particle type")
XNB_DECLARE_FIELD(double                   ,rx   ,"particle position X")
XNB_DECLARE_FIELD(double                   ,ry   ,"particle position Y")
XNB_DECLARE_FIELD(double                   ,rz   ,"particle position Z")
XNB_DECLARE_FIELD(double                   ,vx   ,"particle velocity X")
XNB_DECLARE_FIELD(double                   ,vy   ,"particle velocity Y")
XNB_DECLARE_FIELD(double                   ,vz   ,"particle velocity Z")
XNB_DECLARE_FIELD(double                   ,ax   ,"particle acceleration X")
XNB_DECLARE_FIELD(double                   ,ay   ,"particle acceleration Y")
XNB_DECLARE_FIELD(double                   ,az   ,"particle acceleration Z")

// usefull aliases, often acceleration and force use the same fields,
// acceleration is 'just' divided by mass at some point to turn it to force, or vice-versa
XNB_DECLARE_ALIAS( fx, ax )
XNB_DECLARE_ALIAS( fy, ay )
XNB_DECLARE_ALIAS( fz, az )

// Generic fields, several can be instanciated with different names
XNB_DECLARE_DYNAMIC_FIELD(double               , generic_real , "Generic Scalar" )
XNB_DECLARE_DYNAMIC_FIELD(::onika::math::Vec3d , generic_vec3 , "Generic Vec3d" )
XNB_DECLARE_DYNAMIC_FIELD(::onika::math::Mat3d , generic_mat3 , "Generic Mat3d" )

// unused fields, for compatibility only
struct unused_field_type {};

// utility type to tag fields not available in particular cases
namespace exanb
{
  struct unused_field_id_t {};
  static inline constexpr unused_field_id_t unused_field_id_v = {};
  using default_available_field_sets_t = FieldSets< FieldSet<field::_fx,field::_fy,field::_fz,field::_vx,field::_vy,field::_vz,field::_id,field::_type> >;
  static inline constexpr default_available_field_sets_t default_available_field_sets_v = {};
}

#define XNB_HAS_GRID_FIELDS_DEFINTIONS 1

#ifdef XNB_DOMAIN_SPECIFIC_FIELDS_INCLUDE
#include XNB_DOMAIN_SPECIFIC_FIELDS_INCLUDE
#endif

#ifndef HAS_POSITION_BACKUP_FIELDS
#define HAS_POSITION_BACKUP_FIELDS false
#endif

#ifndef PositionBackupFieldX
#define PositionBackupFieldX ::exanb::unused_field_id_v
#endif

#ifndef PositionBackupFieldY
#define PositionBackupFieldY ::exanb::unused_field_id_v
#endif

#ifndef PositionBackupFieldZ
#define PositionBackupFieldZ ::exanb::unused_field_id_v
#endif

#ifndef XNB_AVAILABLE_FIELD_SETS
#define XNB_AVAILABLE_FIELD_SETS ::exanb::default_available_field_sets_v
#endif

