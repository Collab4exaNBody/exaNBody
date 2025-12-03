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
#include <onika/cuda/cuda_math.h>
#include <cmath>

namespace md
{
  using namespace exanb;

  /* ----------------------------------------------------------------------
     compute Bi by summing conj(Ui)*Zi
  ------------------------------------------------------------------------- */
  template<class ZiRealT, class UiRealT, class BZeroRealT, class BListRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_bi( // READ ONLY
                                      int nelements, int idxz_max, int idxb_max, int idxu_max, int twojmax
//                                    , int const * __restrict__ idxu_block
                                    , int const * __restrict__ idxz_block
                                    , SnapInternal::SNA_ZINDICES const * __restrict__ idxz
                                    , SnapInternal::SNA_BINDICES const * __restrict__ idxb
                                    , ZiRealT const * __restrict__ zlist_r
                                    , ZiRealT const * __restrict__ zlist_i
                                    , UiRealT const * __restrict__ ulisttot_r
                                    , UiRealT const * __restrict__ ulisttot_i
                                    , BZeroRealT const * __restrict__ bzero
                                    , bool bzero_flag, bool wselfall_flag
                                      // WRITE ONLY
                                    , BListRealT * __restrict__ blist
                                      // ORIGINAL PARAMETERS
                                    , int ielem)
  {
    // for j1 = 0,...,twojmax
    //   for j2 = 0,twojmax
    //     for j = |j1-j2|,Min(twojmax,j1+j2),2
    //        b(j1,j2,j) = 0
    //        for mb = 0,...,jmid
    //          for ma = 0,...,j
    //            b(j1,j2,j) +=
    //              2*Conj(u(j,ma,mb))*z(j1,j2,j,ma,mb)

    int itriple = 0;
    int idouble = 0;
    for (int elem1 = 0; elem1 < nelements; elem1++)
      for (int elem2 = 0; elem2 < nelements; elem2++) {

        //double const *zptr_r = &ZLIST_R(idouble*idxz_max);
        //double const *zptr_i = &ZLIST_I(idouble*idxz_max);

        for (int elem3 = 0; elem3 < nelements; elem3++) {
          for (int jjb = 0; jjb < idxb_max; jjb++) {
            const int j1 = IDXB(jjb).j1;
            const int j2 = IDXB(jjb).j2;
            const int j = IDXB(jjb).j;

            int jjz = IDXZ_BLOCK(j1,j2,j);
            int jju = IDXU_BLOCK(j);
            BListRealT sumzu = 0.0;
            for (int mb = 0; 2 * mb < j; mb++)
              for (int ma = 0; ma <= j; ma++) {
                sumzu += ULISTTOT_R(elem3*idxu_max+jju) * ZLIST_R(idouble*idxz_max+jjz) +
                         ULISTTOT_I(elem3*idxu_max+jju) * ZLIST_I(idouble*idxz_max+jjz);
                jjz++;
                jju++;
              } // end loop over ma, mb

            // For j even, handle middle column

            if (j % 2 == 0) {
              int mb = j / 2;
              for (int ma = 0; ma < mb; ma++) {
                sumzu += ULISTTOT_R(elem3*idxu_max+jju) * ZLIST_R(idouble*idxz_max+jjz) +
                         ULISTTOT_I(elem3*idxu_max+jju) * ZLIST_I(idouble*idxz_max+jjz);
                jjz++;
                jju++;
              }

              sumzu += 0.5 * (ULISTTOT_R(elem3*idxu_max+jju) * ZLIST_R(idouble*idxz_max+jjz) +
                              ULISTTOT_I(elem3*idxu_max+jju) * ZLIST_I(idouble*idxz_max+jjz));
            } // end if jeven

            BLIST(itriple*idxb_max+jjb) = 2.0 * sumzu;

          }
          itriple++;
        }
        idouble++;
      }

    // apply bzero shift

    if (bzero_flag) {
      if (!wselfall_flag) {
        itriple = (ielem*nelements+ielem)*nelements+ielem;
        for (int jjb = 0; jjb < idxb_max; jjb++) {
          const int j = IDXB(jjb).j;
          BLIST(itriple*idxb_max+jjb) -= BZERO(j);
        } // end loop over JJ
      } else {
        int itriple = 0;
        for (int elem1 = 0; elem1 < nelements; elem1++)
          for (int elem2 = 0; elem2 < nelements; elem2++) {
            for (int elem3 = 0; elem3 < nelements; elem3++) {
              for (int jjb = 0; jjb < idxb_max; jjb++) {
                const int j = IDXB(jjb).j;
                BLIST(itriple*idxb_max+jjb) -= BZERO(j);
              } // end loop over JJ
              itriple++;
            } // end loop over elem3
          } // end loop over elem1,elem2
      }
    }
  }

}
