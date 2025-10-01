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

#include <md/snap/snap_compute_buffer.h>
#include <md/snap/snap_add_functors.h>

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <cmath>

#ifdef SNAP_AUTOGEN_COMPLEX_MATH
#include <md/snap/snap_math.h>
#endif

namespace md
{
  using namespace exanb;

  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_uarraytot_zero( int nelements
                                        , int idxu_max
                                        , RealT * __restrict__ ulisttot_r
                                        , RealT * __restrict__ ulisttot_i
                                          // OPTIONAL PARAMETERS FOR THREAD TEAM COLLABORATION
                                        , int THREAD_IDX = 0
                                        , int BLOCK_SIZE = 1
                                        )
  {
    const int N = idxu_max * nelements;
    for(int i=THREAD_IDX ; i<N ; i+=BLOCK_SIZE)
    {
      ULISTTOT_R(i) = 0.0;
      ULISTTOT_I(i) = 0.0;
    }
  }


  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_uarraytot_init_wself( // READ ONLY
                                          int nelements
                                        , int twojmax
                                        , int idxu_max
                                        , RealT wself
                                        , bool wselfall_flag
                                          // WRITE ONLY
                                        , RealT * __restrict__ ulisttot_r
                                        , RealT * __restrict__ ulisttot_i
                                          // ORIGINAL PARAMETERS
                                        , int ielem
                                          // OPTIONAL PARAMETERS FOR THREAD TEAM COLLABORATION
                                        , int THREAD_IDX = 0
                                        , int BLOCK_SIZE = 1
                                        ) 
  {
    for (int jelem = 0; jelem < nelements; jelem++)
    {
      for (int j=THREAD_IDX; j<=twojmax; j+=BLOCK_SIZE )
      {
        int jju = IDXU_BLOCK(j);
        for (int mb = 0; mb <= j; mb++)
        {
          for (int ma = 0; ma <= j; ma++)
          {
            // utot(j,ma,ma) = wself, sometimes
            if( (jelem == ielem || wselfall_flag) && (ma==mb) ) ULISTTOT_R(jelem*idxu_max+jju) = wself; ///// double check this
            jju++;
          }
        }
      }
    }
  }


  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_zero_uarraytot_with_wself( // READ ONLY
                                          int nelements
                                        , int twojmax
                                        , int idxu_max
                                        , RealT wself
                                        , bool wselfall_flag
                                          // WRITE ONLY
                                        , RealT * __restrict__ ulisttot_r
                                        , RealT * __restrict__ ulisttot_i
                                          // ORIGINAL PARAMETERS
                                        , int ielem ) 
  {
    snap_uarraytot_zero( nelements, idxu_max, ulisttot_r, ulisttot_i );
    snap_uarraytot_init_wself( nelements, twojmax, idxu_max, wself, wselfall_flag, ulisttot_r, ulisttot_i, ielem );
  }

  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  inline void snap_compute_uarray( // READ ONLY
                                   int twojmax
                                 , RealT const * __restrict__ rootpqarray
                                   // WRITE ONLY
                                 , RealT * __restrict__ ulist_r_ij_jj // = ulist_r_ij + jj offset = ULIST_R_J(jj), accessed through ULIST_R_JJ(i)
                                 , RealT * __restrict__ ulist_i_ij_jj // = ulist_i_ij_jj + jj offset, accessed with ULIST_I_JJ(i)
                                   // ORIGINAL PARAMETERS
                                 , RealT x, RealT y, RealT z, RealT z0, RealT r )
  {
    RealT r0inv;
    RealT a_r, b_r, a_i, b_i;
    RealT rootpq;

    // compute Cayley-Klein parameters for unit quaternion

    r0inv = 1.0 / sqrt(r * r + z0 * z0);
    a_r = r0inv * z0;
    a_i = -r0inv * z;
    b_r = r0inv * y;
    b_i = -r0inv * x;

    // VMK Section 4.8.2
    //double* ulist_r = ulist_r_ij[jj];
    //double* ulist_i = ulist_i_ij[jj];

    ULIST_R_JJ(0) = 1.0;
    ULIST_I_JJ(0) = 0.0;

    for (int j = 1; j <= twojmax; j++) {
      int jju = IDXU_BLOCK(j);
      int jjup = IDXU_BLOCK(j-1);

      // fill in left side of matrix layer from previous layer

      for (int mb = 0; 2*mb <= j; mb++) {
        ULIST_R_JJ(jju) = 0.0;
        ULIST_I_JJ(jju) = 0.0;

        for (int ma = 0; ma < j; ma++) {
          rootpq = ROOTPQARRAY(j - ma,j - mb);
          ULIST_R_JJ(jju) += rootpq * (a_r * ULIST_R_JJ(jjup) + a_i * ULIST_I_JJ(jjup));
          ULIST_I_JJ(jju) += rootpq * (a_r * ULIST_I_JJ(jjup) - a_i * ULIST_R_JJ(jjup));

          rootpq = ROOTPQARRAY(ma + 1,j - mb);
          ULIST_R_JJ(jju+1) = -rootpq * (b_r * ULIST_R_JJ(jjup) + b_i * ULIST_I_JJ(jjup));
          ULIST_I_JJ(jju+1) = -rootpq * (b_r * ULIST_I_JJ(jjup) - b_i * ULIST_R_JJ(jjup));
          jju++;
          jjup++;
        }
        jju++;
      }

      // copy left side to right side with inversion symmetry VMK 4.4(2)
      // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])

      jju = IDXU_BLOCK(j);
      jjup = jju+(j+1)*(j+1)-1;
      int mbpar = 1;
      for (int mb = 0; 2*mb <= j; mb++) {
        int mapar = mbpar;
        for (int ma = 0; ma <= j; ma++) {
          if (mapar == 1) {
            ULIST_R_JJ(jjup) = ULIST_R_JJ(jju);
            ULIST_I_JJ(jjup) = -ULIST_I_JJ(jju);
          } else {
            ULIST_R_JJ(jjup) = -ULIST_R_JJ(jju);
            ULIST_I_JJ(jjup) = ULIST_I_JJ(jju);
          }
          mapar = -mapar;
          jju++;
          jjup--;
        }
        mbpar = -mbpar;
      }
    }
  }

  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline RealT snap_compute_sfac( // READ ONLY
                                          RealT rmin0, bool switch_flag, bool switch_inner_flag
                                          // ORIGINAL PARAMTERS
                                        , RealT r, RealT rcut, RealT sinner, RealT dinner)
  {
    RealT sfac = 0.0;

    // calculate sfac = sfac_outer

    if (switch_flag == 0) sfac = 1.0;
    else if (r <= rmin0) sfac = 1.0;
    else if (r > rcut) sfac = 0.0;
    else {
      RealT rcutfac = M_PI / (rcut - rmin0);
      sfac = 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }

    // calculate sfac *= sfac_inner, rarely visited

    if (switch_inner_flag == 1 && r < sinner + dinner) {
      if (r > sinner - dinner) {
        RealT rcutfac = (M_PI/2) / dinner;
        sfac *= 0.5 * (1.0 - cos( (M_PI/2) + (r - sinner) * rcutfac));
      } else sfac = 0.0;
    }

    return sfac;
  }

  template<class RealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_add_uarraytot( // READ ONLY
                                         int twojmax
                                       , int jelem
                                       , int idxu_max
                                       , RealT sfac_wj
                                       , RealT const * __restrict__ ulist_r_ij_jj
                                       , RealT const * __restrict__ ulist_i_ij_jj
                                         // WRITE ONLY
                                       , RealT * __restrict__ ulisttot_r
                                       , RealT * __restrict__ ulisttot_i )
  {
    // const int N = SUM_INT_SQR(twojmax+1); // = idxu_max
    for (int jju = 0; jju < idxu_max ; ++jju)
    {
      ULISTTOT_R(jelem*idxu_max+jju) += sfac_wj * ULIST_R_JJ(jju);
      ULISTTOT_I(jelem*idxu_max+jju) += sfac_wj * ULIST_I_JJ(jju);
    }
  }

  template<class SnapXSForceExtStorageT, class AccumFuncT = SimpleAccumFunctor>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_add_nbh_contrib_to_uarraytot(
                                   int twojmax
                                 , double sfac_wj, double x, double y, double z, double z0, double r
                                 , double const * __restrict__ rootpqarray
                                 , double * __restrict__ ulisttot_r 
                                 , double * __restrict__ ulisttot_i
                                 , SnapXSForceExtStorageT& ext
                                 , AccumFuncT merge_func = {} )
  {
    const int idxu_max = SUM_INT_SQR(twojmax+1);
    snap_compute_uarray( twojmax, rootpqarray, ext.m_U_array.r(), ext.m_U_array.i(), x,y,z,z0,r );
    snap_add_uarraytot( twojmax, 0, idxu_max, sfac_wj, ext.m_U_array.r(), ext.m_U_array.i(), ulisttot_r, ulisttot_i );
  }

  template<class RealT, class SnapXSForceExtStorageT,int twojmax, class AccumFuncT = SimpleAccumFunctor>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_add_nbh_contrib_to_uarraytot(
                                   onika::IntConst<twojmax> _twojmax_
                                 , RealT sfac_wj, RealT x, RealT y, RealT z, RealT z0, RealT r
                                 , RealT const * __restrict__ rootpqarray
                                 , RealT * __restrict__ ulisttot_r 
                                 , RealT * __restrict__ ulisttot_i
                                 , SnapXSForceExtStorageT& ext
                                 , AccumFuncT merge_func = {} )
  {  
    const RealT r0inv = 1.0 / sqrt(r * r + z0 * z0);
    const RealT a_r = r0inv * z0;
    const RealT a_i = -r0inv * z;
    const RealT b_r = r0inv * y;
    const RealT b_i = -r0inv * x;

#   define CONST_DECLARE(var,value) static constexpr RealT var = value

#   ifdef SNAP_AUTOGEN_COMPLEX_MATH

      // version using Complex and Complex3D arithmetic
    using namespace SnapMath;
    const SnapMath::Complexd a = { a_r , a_i };
    const SnapMath::Complexd b = { b_r , b_i };
    [[maybe_unused]] static constexpr SnapMath::Complexd U_ZERO = {0.,0.};
    [[maybe_unused]] static constexpr SnapMath::Complexd U_UNIT = {1.,0.};
#   define U_DECLARE(var) SnapMath::Complexd var
#   define BAKE_U_BLEND(c,var) const auto c##_##var = ( conj(c) * var )
#   define U_BLEND(c,var) c##_##var
#   define U_ASSIGN(var,expr) var = expr
#   define U_STORE(var,jju) merge_func( ULISTTOT_R(jju) , sfac_wj * var.r ) ; merge_func( ULISTTOT_I(jju) , sfac_wj * var.i )

#   else

    // version that uses only scalar arithmetic
    [[maybe_unused]] static constexpr RealT U_ZERO_R = 0.;
    [[maybe_unused]] static constexpr RealT U_ZERO_I = 0.;
    [[maybe_unused]] static constexpr RealT U_UNIT_R = 1.;
    [[maybe_unused]] static constexpr RealT U_UNIT_I = 0.;
#   define U_DECLARE(var) RealT var##_r , var##_i
#   define BAKE_U_BLEND(c,var) const RealT c##_##var##_r = c##_r * var##_r + c##_i * var##_i , c##_##var##_i = c##_r * var##_i - c##_i * var##_r
#   define U_BLEND_R(c,var) c##_##var##_r
#   define U_BLEND_I(c,var) c##_##var##_i
#   define CONJ_R(r_expr) (r_expr)
#   define CONJ_U_R(var) var##_r
#   define CONJ_I(i_expr) -(i_expr)
#   define CONJ_U_I(var) -var##_i
#   define U_ASSIGN_R(var,expr) var##_r = expr
#   define U_ASSIGN_I(var,expr) var##_i = expr
#   define U_STORE(var,jju) merge_func( ULISTTOT_R(jju) , sfac_wj * var##_r ) ; merge_func( ULISTTOT_I(jju) , sfac_wj * var##_i )

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

  }

}
