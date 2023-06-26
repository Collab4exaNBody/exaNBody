#pragma once

#include <onika/soatl/field_id.h>

#define SOATL_DECLARE_FIELD(__type,__name,__desc) \
struct __##__name {}; \
namespace onika { namespace soatl { template<> struct FieldId<__##__name> { \
    using value_type = __type; \
    using Id = __##__name; \
    static const char* name() { return __desc ; } \
  }; } } \
::onika::soatl::FieldId<__##__name> __name;

SOATL_DECLARE_FIELD(double	,particle_rx	,"Particle position X");
SOATL_DECLARE_FIELD(double	,particle_ry	,"Particle position Y");
SOATL_DECLARE_FIELD(double	,particle_rz	,"Particle position Z");
SOATL_DECLARE_FIELD(unsigned char,particle_atype,"Particle atom type");
SOATL_DECLARE_FIELD(double	,particle_e	,"Particle energy");
SOATL_DECLARE_FIELD(int32_t	,particle_mid	,"Particle molecule id");
SOATL_DECLARE_FIELD(float	,particle_dist	,"Particle pair distance");
SOATL_DECLARE_FIELD(int16_t	,particle_tmp1	,"Particle Temporary 1");
SOATL_DECLARE_FIELD(int8_t	,particle_tmp2	,"Particle Temporary 2");

SOATL_DECLARE_FIELD(float	,particle_rx_f	,"Particle position X (single precision)");
SOATL_DECLARE_FIELD(float	,particle_ry_f	,"Particle position Y (single precision)");
SOATL_DECLARE_FIELD(float	,particle_rz_f	,"Particle position Z (single precision)");
SOATL_DECLARE_FIELD(float	,particle_e_f	,"Particle energy (single precision)");

#undef SOATL_DECLARE_FIELD
