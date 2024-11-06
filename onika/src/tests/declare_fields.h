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
