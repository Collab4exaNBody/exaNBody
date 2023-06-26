#pragma once

#include <onika/soatl/field_id.h>
#include <cstdint>

#define XSTAMP_VAR_USED(x) inline decltype(x) _disable_unused_warning_##x() { return x; }

#define XSTAMP_DECLARE_FIELD(__type,__name,__desc) \
namespace exanb { namespace field { struct _##__name {}; } } \
namespace onika { namespace soatl { template<> struct FieldId<::exanb::field::_##__name> { \
    using value_type = __type; \
    using Id = ::exanb::field::_##__name; \
    static inline const char* short_name() { return #__name ; } \
    static inline const char* name() { return __desc ; } \
  }; } } \
namespace exanb { namespace field { static constexpr ::onika::soatl::FieldId<_##__name> __name; XSTAMP_VAR_USED(__name) /*inline void _disable_unused_warning_##__name(){ ::soatl::FieldId<_##__name> x=__name; x=x; }*/ } }

#define XSTAMP_DECLARE_ALIAS(A,F) namespace exanb { namespace field { using _##A=_##F; static constexpr ::onika::soatl::FieldId<_##A> A; } }

