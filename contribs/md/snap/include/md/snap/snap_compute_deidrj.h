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
     compute dEidRj
  ------------------------------------------------------------------------- */
  template<class UiRealT, class YiRealT, class ForceRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_deidrj( // READ ONLY
                                          int elem_duarray, int twojmax, int idxu_max
//                                        , int const * __restrict__ idxu_block
                                        , UiRealT const * __restrict__ dulist_r
                                        , UiRealT const * __restrict__ dulist_i
                                        , YiRealT const * __restrict__ ylist_r
                                        , YiRealT const * __restrict__ ylist_i
                                          // ORIGINAL PARAMETERS
                                        , ForceRealT * __restrict__ dedr)
  {

    for (int k = 0; k < 3; k++)
      dedr[k] = 0.0;

    int jelem = elem_duarray;
    for (int j = 0; j <= twojmax; j++) {
      int jju = IDXU_BLOCK(j);

      for (int mb = 0; 2*mb < j; mb++)
        for (int ma = 0; ma <= j; ma++) {

          //double const * dudr_r = dulist_r[jju];
          //double const * dudr_i = dulist_i[jju];
          YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max+jju);
          YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max+jju);

          for (int k = 0; k < 3; k++)
            dedr[k] +=
              DULIST_R(jju,k) * jjjmambyarray_r +
              DULIST_I(jju,k) * jjjmambyarray_i;
          jju++;
        } //end loop over ma mb

      // For j even, handle middle column

      if (j%2 == 0) {

        int mb = j/2;
        for (int ma = 0; ma < mb; ma++) {
          //double const * dudr_r = dulist_r[jju];
          //double const * dudr_i = dulist_i[jju];
          YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max+jju);
          YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max+jju);

          for (int k = 0; k < 3; k++)
            dedr[k] +=
              DULIST_R(jju,k) * jjjmambyarray_r +
              DULIST_I(jju,k) * jjjmambyarray_i;
          jju++;
        }

        //double const * dudr_r = dulist_r[jju];
        //double const * dudr_i = dulist_i[jju];
        YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max+jju);
        YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max+jju);

        for (int k = 0; k < 3; k++)
          dedr[k] +=
            (DULIST_R(jju,k) * jjjmambyarray_r +
             DULIST_I(jju,k) * jjjmambyarray_i)*0.5;
        // jju++;

      } // end if jeven

    } // end loop over j

    for (int k = 0; k < 3; k++)
      dedr[k] *= 2.0;

  }



  template<class UiRealT, class YiRealT, class ForceRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_deidrj_alt( // READ ONLY
                                          int elem_duarray, int twojmax, int idxu_max
                                        , int const * __restrict__ y_jju_map, int idxu_max_alt
                                        , UiRealT const * __restrict__ dulist_r
                                        , UiRealT const * __restrict__ dulist_i
                                        , YiRealT const * __restrict__ ylist_r
                                        , YiRealT const * __restrict__ ylist_i
                                          // ORIGINAL PARAMETERS
                                        , ForceRealT * __restrict__ dedr)
  {

    for (int k = 0; k < 3; k++)
      dedr[k] = 0.0;

    int jelem = elem_duarray;
    for (int j = 0; j <= twojmax; j++) {
      int jju = IDXU_BLOCK(j);

      for (int mb = 0; 2*mb < j; mb++)
        for (int ma = 0; ma <= j; ma++) {

          //double const * dudr_r = dulist_r[jju];
          //double const * dudr_i = dulist_i[jju];
          int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert( jju_map != -1 && jju_map <= jju );
          YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max_alt+jju_map);
          YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max_alt+jju_map);

          for (int k = 0; k < 3; k++)
            dedr[k] +=
              DULIST_R(jju,k) * jjjmambyarray_r +
              DULIST_I(jju,k) * jjjmambyarray_i;
          jju++;
        } //end loop over ma mb

      // For j even, handle middle column

      if (j%2 == 0) {

        int mb = j/2;
        for (int ma = 0; ma < mb; ma++) {
          //double const * dudr_r = dulist_r[jju];
          //double const * dudr_i = dulist_i[jju];
          int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert( jju_map != -1 && jju_map <= jju );
          YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max_alt+jju_map);
          YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max_alt+jju_map);

          for (int k = 0; k < 3; k++)
            dedr[k] +=
              DULIST_R(jju,k) * jjjmambyarray_r +
              DULIST_I(jju,k) * jjjmambyarray_i;
          jju++;
        }

        //double const * dudr_r = dulist_r[jju];
        //double const * dudr_i = dulist_i[jju];
        int jju_map=jju; if(y_jju_map!=nullptr) jju_map=y_jju_map[jju]; assert( jju_map != -1 && jju_map <= jju );
        YiRealT jjjmambyarray_r = YLIST_R(jelem*idxu_max_alt+jju_map);
        YiRealT jjjmambyarray_i = YLIST_I(jelem*idxu_max_alt+jju_map);

        for (int k = 0; k < 3; k++)
          dedr[k] +=
            (DULIST_R(jju,k) * jjjmambyarray_r +
             DULIST_I(jju,k) * jjjmambyarray_i)*0.5;
        // jju++;

      } // end if jeven

    } // end loop over j

    for (int k = 0; k < 3; k++)
      dedr[k] *= 2.0;

  }



}
