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

#include <onika/cuda/cuda.h>
#include <cmath>

// tells if we use Complex arithmetic classes or unroll all to scalar expressions
//#define SNAP_AUTOGEN_COMPLEX_MATH 1

#include <md/snap/snap_compute_ui.h>
#include <md/snap/snap_compute_duidrj.h>
#include <md/snap/snap_compute_deidrj.h>

#ifdef SNAP_AUTOGEN_COMPLEX_MATH
#include <md/snap/snap_math.h>
#endif

namespace md
{
  using namespace exanb;

  template<class RijRealT, class RootPQRealT, class ForceRealT, class SnapXSForceExtStorageT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void add_nbh_contrib_to_force(
                 int twojmax, int idxu_max, int jelem
               , RijRealT wj_jj, RijRealT rcutij_jj, RijRealT sinnerij_jj, RijRealT dinnerij_jj
               , RijRealT x, RijRealT y, RijRealT z, RijRealT z0, RijRealT r, RijRealT rsq
               , RootPQRealT const * __restrict__ rootpqarray
               , int const * __restrict__ y_jju_map, int idxu_max_alt
               , RijRealT rmin0, RijRealT rfac0, bool switch_flag, bool switch_inner_flag, bool chem_flag
               , ForceRealT * __restrict__ fij
               , SnapXSForceExtStorageT& ext )
  {
//    printf("generic add_nbh_contrib_to_force\n");
    // here we use UiTot as scratch buffer to compute Uarray for jj
    snap_compute_uarray( twojmax, rootpqarray, ext.m_U_array.r(), ext.m_U_array.i(), x, y, z, z0, r );

    // reads ulist, sinner and dinner
    // writes dulist
    snap_compute_duidrj( twojmax, idxu_max, x,y,z, rcutij_jj , wj_jj
                       , ext.m_U_array.r(), ext.m_U_array.i()
                       , rootpqarray
                       , sinnerij_jj, dinnerij_jj
                       , rmin0, rfac0, switch_flag, switch_inner_flag, chem_flag                             
                       , ext.m_DU_array.r(), ext.m_DU_array.i() // OUTPUTS
                       );

    // reads dulist and ylist
    // writes final force
    snap_compute_deidrj_alt( jelem, twojmax, idxu_max
                       , y_jju_map, idxu_max_alt
                       , ext.m_DU_array.r(), ext.m_DU_array.i()
                       , ext.m_Y_array.r(), ext.m_Y_array.i()
                       , fij );
  }

  template<int twojmax, class RijRealT, class RootPQRealT, class ForceRealT, class SnapXSForceExtStorageT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void add_nbh_contrib_to_force(
                 onika::IntConst<twojmax> _twojmax_, int idxu_max, int jelem
               , RijRealT wj_jj, RijRealT rcutij_jj, RijRealT sinnerij_jj, RijRealT dinnerij_jj
               , RijRealT x, RijRealT y, RijRealT z, RijRealT z0, RijRealT r, RijRealT rsq
               , RootPQRealT const * __restrict__ rootpqarray
               , int const * __restrict__ y_jju_map, int idxu_max_alt
               , RijRealT rmin0, RijRealT rfac0, bool switch_flag, bool switch_inner_flag, bool chem_flag
               , ForceRealT * __restrict__ fij
               , SnapXSForceExtStorageT& ro_ext )
  {
    using UiRealT = typename SnapXSForceExtStorageT::BufferRealType;
    
//    printf("specific add_nbh_contrib_to_force\n");
    const SnapXSForceExtStorageT& ext = ro_ext;

    const RijRealT r0inv = static_cast<RijRealT>(1.0) / sqrt(r * r + z0 * z0);
    const RijRealT a_r = r0inv * z0;
    const RijRealT a_i = -r0inv * z;
    const RijRealT b_r = r0inv * y;
    const RijRealT b_i = -r0inv * x;

    const RijRealT rscale0 = rfac0 * M_PI / (rcutij_jj - rmin0);
    const RijRealT dz0dr = z0 / r - (r*rscale0) * (rsq + z0 * z0) / rsq;

    RijRealT rinv = 1.0 / r;
    RijRealT ux = x * rinv;
    RijRealT uy = y * rinv;
    RijRealT uz = z * rinv;

    const RijRealT dr0invdr = -pow(r0inv, 3) * (r + z0 * dz0dr);

    const RijRealT dr0inv[3] = { dr0invdr * ux , dr0invdr * uy , dr0invdr * uz };
    // dr0inv[0] = dr0invdr * ux;
    // dr0inv[1] = dr0invdr * uy;
    // dr0inv[2] = dr0invdr * uz;

    const RijRealT dz0[3] = { dz0dr * ux , dz0dr * uy , dz0dr * uz };
    // dz0[0] = dz0dr * ux;
    // dz0[1] = dz0dr * uy;
    // dz0[2] = dz0dr * uz;

    RijRealT da_r[3] , da_i[3] , db_r[3] , db_i[3];
    for (int k = 0; k < 3; k++) {
      da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
      da_i[k] = -z * dr0inv[k];
    }
    da_i[2] += -r0inv;
    for (int k = 0; k < 3; k++) {
      db_r[k] = y * dr0inv[k];
      db_i[k] = -x * dr0inv[k];
    }
    db_i[0] += -r0inv;
    db_r[1] += r0inv;

    const RijRealT sfac  = wj_jj * snap_compute_sfac ( rmin0, switch_flag, switch_inner_flag, r, rcutij_jj, sinnerij_jj, dinnerij_jj );
    const RijRealT dsfac = wj_jj * snap_compute_dsfac( rmin0, switch_flag, switch_inner_flag, r, rcutij_jj, sinnerij_jj, dinnerij_jj );

    const RijRealT da_x_r = da_r[0];
    const RijRealT da_x_i = da_i[0];
    
    const RijRealT da_y_r = da_r[1];
    const RijRealT da_y_i = da_i[1];

    const RijRealT da_z_r = da_r[2];
    const RijRealT da_z_i = da_i[2];


    const RijRealT db_x_r = db_r[0];
    const RijRealT db_x_i = db_i[0];
    
    const RijRealT db_y_r = db_r[1];
    const RijRealT db_y_i = db_i[1];

    const RijRealT db_z_r = db_r[2];
    const RijRealT db_z_i = db_i[2];

    const auto * __restrict__ Y_r = ext.m_Y_array.r() + jelem * idxu_max_alt;
    const auto * __restrict__ Y_i = ext.m_Y_array.i() + jelem * idxu_max_alt;

#   define CONST_DECLARE(var,value) static constexpr RootPQRealT var = static_cast<RootPQRealT>(value)

#   ifdef SNAP_AUTOGEN_COMPLEX_MATH

    using namespace SnapMath;    
    const SnapMath::Complexd a = { a_r , a_i };
    const SnapMath::Complexd b = { b_r , b_i };
    const SnapMath::Complexd da_x = {da_x_r,da_x_i};
    const SnapMath::Complexd da_y = {da_y_r,da_y_i};
    const SnapMath::Complexd da_z = {da_z_r,da_z_i};
    const SnapMath::Complexd db_x = {db_x_r,db_x_i};
    const SnapMath::Complexd db_y = {db_y_r,db_y_i};
    const SnapMath::Complexd db_z = {db_z_r,db_z_i};    
    const SnapMath::Complex3d da = { da_x , da_y , da_z };
    const SnapMath::Complex3d db = { db_x , db_y , db_z };
    const SnapMath::Double3d u = { ux , uy , uz };
    static constexpr SnapMath::Complexd U_ZERO = {0.,0.};
    static constexpr SnapMath::Complexd U_UNIT = {1.,0.};
    static constexpr SnapMath::Complex3d dU_ZERO = { U_ZERO , U_ZERO , U_ZERO };

#   define U_DECLARE(var)          SnapMath::Complexd var
#   define BAKE_U_BLEND(c,var)     const auto c##_##var = ( conj(c) * var ); \
                                   const auto d##c##_##var = ( conj(d##c) * var + conj(c) * d##var )
#   define U_BLEND(c,var)          c##_##var
#   define U_ASSIGN(var,expr)      var = expr

#   define dU_BLEND(c,var) d##c##_##var
#   define dU_POSTPROCESS(var,jju) d##var = dsfac * var * u + sfac * d##var

#   define CONJC3D(var) conj(var)
#   define dU_DECLARE(var) SnapMath::Complex3d d##var
#   define dU_ASSIGN(var,expr) d##var = expr
#   define dU_STORE_FSKIP(var,xxx,jju_map) /**/
#   define dU_STORE(var,xxx,jju_map)       /*do{*/ /*int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert(jju_map==cjjumap);*/ \
      fij[0] +=      d##var.x.r * Y_r[jju_map] + d##var.x.i * Y_i[jju_map];  \
      fij[1] +=      d##var.y.r * Y_r[jju_map] + d##var.y.i * Y_i[jju_map];  \
      fij[2] +=      d##var.z.r * Y_r[jju_map] + d##var.z.i * Y_i[jju_map] //; }while(0)
#   define dU_STORE_FHALF(var,xxx,jju_map) /*do{*/ /*int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert(jju_map==cjjumap);*/ \
      fij[0] += 0.5*(d##var.x.r * Y_r[jju_map] + d##var.x.i * Y_i[jju_map]); \
      fij[1] += 0.5*(d##var.y.r * Y_r[jju_map] + d##var.y.i * Y_i[jju_map]); \
      fij[2] += 0.5*(d##var.z.r * Y_r[jju_map] + d##var.z.i * Y_i[jju_map]) //; }while(0)

#   else

    static constexpr UiRealT U_UNIT_R = static_cast<UiRealT>(1.);
    static constexpr UiRealT U_UNIT_I = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_X_R = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_X_I = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_Y_R = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_Y_I = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_Z_R = static_cast<UiRealT>(0.);
    static constexpr UiRealT dU_ZERO_Z_I = static_cast<UiRealT>(0.);

#   define U_DECLARE(var)       UiRealT var##_r , var##_i
#   define dU_DECLARE(var)      UiRealT d##var##_x##_r , d##var##_x##_i , d##var##_y##_r , d##var##_y##_i , d##var##_z##_r , d##var##_z##_i

#   define BAKE_U_BLEND(c,var)  const UiRealT c##_##var##_r = c##_r * var##_r + c##_i * var##_i , \
                                              c##_##var##_i = c##_r * var##_i - c##_i * var##_r ; \
                                const UiRealT d##c##_d##var##_x##_r = d##c##_x##_r * var##_r + d##c##_x##_i * var##_i \
                                                                    + c##_r * d##var##_x##_r + c##_i * d##var##_x##_i; \
                                const UiRealT d##c##_d##var##_x##_i = d##c##_x##_r * var##_i - d##c##_x##_i * var##_r \
                                                                    + c##_r * d##var##_x##_i - c##_i * d##var##_x##_r; \
                                const UiRealT d##c##_d##var##_y##_r = d##c##_y##_r * var##_r + d##c##_y##_i * var##_i \
                                                                    + c##_r * d##var##_y##_r + c##_i * d##var##_y##_i; \
                                const UiRealT d##c##_d##var##_y##_i = d##c##_y##_r * var##_i - d##c##_y##_i * var##_r \
                                                                    + c##_r * d##var##_y##_i - c##_i * d##var##_y##_r; \
                                const UiRealT d##c##_d##var##_z##_r = d##c##_z##_r * var##_r + d##c##_z##_i * var##_i \
                                                                    + c##_r * d##var##_z##_r + c##_i * d##var##_z##_i; \
                                const UiRealT d##c##_d##var##_z##_i = d##c##_z##_r * var##_i - d##c##_z##_i * var##_r \
                                                                    + c##_r * d##var##_z##_i - c##_i * d##var##_z##_r
#   define U_BLEND_R(c,var)     c##_##var##_r
#   define U_BLEND_I(c,var)     c##_##var##_i
#   define CONJ_R(r_expr)       (r_expr)
#   define CONJ_U_R(var)        var##_r
#   define CONJ_I(i_expr)       -(i_expr)
#   define CONJ_U_I(var)        -var##_i

#   define U_ASSIGN_R(var,expr) var##_r = expr
#   define U_ASSIGN_I(var,expr) var##_i = expr

#   define CONJC3D_X_R(r_expr) (r_expr)
#   define CONJC3D_dU_X_R(var) var##_x##_r
#   define CONJC3D_X_I(i_expr) -(i_expr)
#   define CONJC3D_dU_X_I(var) -var##_x##_i
#   define CONJC3D_Y_R(r_expr) (r_expr)
#   define CONJC3D_dU_Y_R(var) var##_y##_r
#   define CONJC3D_Y_I(i_expr) -(i_expr)
#   define CONJC3D_dU_Y_I(var) -var##_y##_i
#   define CONJC3D_Z_R(r_expr) (r_expr)
#   define CONJC3D_dU_Z_R(var) var##_z##_r
#   define CONJC3D_Z_I(i_expr) -(i_expr)
#   define CONJC3D_dU_Z_I(var) -var##_z##_i
 
#   define dU_BLEND_X_R(c,var) d##c##_d##var##_x##_r
#   define dU_BLEND_X_I(c,var) d##c##_d##var##_x##_i
#   define dU_BLEND_Y_R(c,var) d##c##_d##var##_y##_r
#   define dU_BLEND_Y_I(c,var) d##c##_d##var##_y##_i
#   define dU_BLEND_Z_R(c,var) d##c##_d##var##_z##_r
#   define dU_BLEND_Z_I(c,var) d##c##_d##var##_z##_i

#   define dU_POSTPROCESS(var,jju) \
        d##var##_x##_r = static_cast<UiRealT>(dsfac) * var##_r * static_cast<UiRealT>(ux) + static_cast<UiRealT>(sfac) * d##var##_x##_r; \
        d##var##_x##_i = static_cast<UiRealT>(dsfac) * var##_i * static_cast<UiRealT>(ux) + static_cast<UiRealT>(sfac) * d##var##_x##_i; \
        d##var##_y##_r = static_cast<UiRealT>(dsfac) * var##_r * static_cast<UiRealT>(uy) + static_cast<UiRealT>(sfac) * d##var##_y##_r; \
        d##var##_y##_i = static_cast<UiRealT>(dsfac) * var##_i * static_cast<UiRealT>(uy) + static_cast<UiRealT>(sfac) * d##var##_y##_i; \
        d##var##_z##_r = static_cast<UiRealT>(dsfac) * var##_r * static_cast<UiRealT>(uz) + static_cast<UiRealT>(sfac) * d##var##_z##_r; \
        d##var##_z##_i = static_cast<UiRealT>(dsfac) * var##_i * static_cast<UiRealT>(uz) + static_cast<UiRealT>(sfac) * d##var##_z##_i

#   define dU_ASSIGN_X_R(var,expr) d##var##_x##_r = expr
#   define dU_ASSIGN_X_I(var,expr) d##var##_x##_i = expr
#   define dU_ASSIGN_Y_R(var,expr) d##var##_y##_r = expr
#   define dU_ASSIGN_Y_I(var,expr) d##var##_y##_i = expr
#   define dU_ASSIGN_Z_R(var,expr) d##var##_z##_r = expr
#   define dU_ASSIGN_Z_I(var,expr) d##var##_z##_i = expr

#   define dU_STORE_FSKIP(var,xxx,jju_map) /**/
#   define dU_STORE(var,xxx,jju_map) /*do{*/ /*int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert(jju_map!=-1);*/ \
      fij[0] +=      d##var##_x##_r * Y_r[jju_map] + d##var##_x##_i * Y_i[jju_map];  \
      fij[1] +=      d##var##_y##_r * Y_r[jju_map] + d##var##_y##_i * Y_i[jju_map];  \
      fij[2] +=      d##var##_z##_r * Y_r[jju_map] + d##var##_z##_i * Y_i[jju_map] //; }while(0)
#   define dU_STORE_FHALF(var,xxx,jju_map) /*do{*/ /*int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert(jju_map!=-1);*/ \
      fij[0] += static_cast<UiRealT>(0.5)*(d##var##_x##_r * Y_r[jju_map] + d##var##_x##_i * Y_i[jju_map]); \
      fij[1] += static_cast<UiRealT>(0.5)*(d##var##_y##_r * Y_r[jju_map] + d##var##_y##_i * Y_i[jju_map]); \
      fij[2] += static_cast<UiRealT>(0.5)*(d##var##_z##_r * Y_r[jju_map] + d##var##_z##_i * Y_i[jju_map]) //; }while(0)

#   endif

    static constexpr int jmax = twojmax/2;
    static_assert( jmax==2 || jmax==3 || jmax==4 );

#   define SNAP_AUTOGEN_NO_UNDEF 1
    if constexpr ( jmax == 2 )
    {
#     include <md/snap/compute_ui_jmax2.hxx>
    }

    if constexpr ( jmax == 3 )
    {
#     include <md/snap/compute_ui_jmax3.hxx>
    }

#   undef SNAP_AUTOGEN_NO_UNDEF
    if constexpr ( jmax == 4 )
    {
#     include <md/snap/compute_ui_jmax4.hxx>
    }

    fij[0] *= static_cast<ForceRealT>(2.);
    fij[1] *= static_cast<ForceRealT>(2.);
    fij[2] *= static_cast<ForceRealT>(2.);
  }

}

