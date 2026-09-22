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
#include <md/snap/snap_compute_ui.h>
#include <md/snap/snap_compute_duidrj.h>

// Mono-element (nelements==1, chem_flag==false) port of LAMMPS ML-SNAP's
// SNA::compute_dbidrj(): derivative of the bispectrum dB(j1,j2,j)/dr_j for one
// neighbor, reusing the atom-level zlist (Z, built once by snap_compute_zi,
// no neighbor dependence) and that neighbor's dulist (dU/dr, from
// snap_compute_duidrj) -- no derivative of Z is ever computed. Each canonical
// triple (j1,j2,j) gets three "leg" contributions, each reusing zlist at a
// permuted triple combined with dulist at a different angular-momentum level:
//
//   dbdr(j1,j2,j) += 2 * sum_{mb<j/2,ma<=j}   dudr(j ,ma,mb) . z(j1,j2,j )   [weight 1]
//   dbdr(j1,j2,j) += 2 * sum_{mb<j1/2,ma<=j1} dudr(j1,ma,mb) . z(j ,j2,j1)  [weight (j+1)/(j1+1)]
//   dbdr(j1,j2,j) += 2 * sum_{mb<j2/2,ma<=j2} dudr(j2,ma,mb) . z(j ,j1,j2)  [weight (j+1)/(j2+1)]
//
// (plus each leg's even-j half-weighted middle row, mirroring snap_compute_bi's
// own mb==j/2 handling). bnorm_flag drops the (j+1)/(j{1,2}+1) rescale, since
// it is already baked into zlist in that mode. Verified against the actual
// LAMMPS ML-SNAP/sna.cpp source (not reconstructed from memory), in particular
// leg 3's z-index order (j,j1,j2), which is easy to get backwards.
namespace md
{
  using namespace exanb;

  // accumulates one leg's sum[0..2] += sum_{mb,ma} dudr(leg_j,ma,mb) . z(leg_j,...)
  // starting at (jjz0,jju0); caller supplies the pre-permuted z/u base offsets.
  template<class ZiRealT, class UiRealT, class DbRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_dbidrj_leg( // READ ONLY
                                      int leg_j, int jjz0, int jju0
                                    , ZiRealT const * __restrict__ zlist_r
                                    , ZiRealT const * __restrict__ zlist_i
                                    , UiRealT const * __restrict__ dulist_r
                                    , UiRealT const * __restrict__ dulist_i
                                      // WRITE ONLY (accumulated into, caller zero-inits)
                                    , DbRealT * __restrict__ sum )
  {
    int jjz = jjz0;
    int jju = jju0;
    for (int mb = 0; 2 * mb < leg_j; mb++)
      for (int ma = 0; ma <= leg_j; ma++) {
        for (int k = 0; k < 3; k++) sum[k] += DULIST_R(jju,k) * ZLIST_R(jjz) + DULIST_I(jju,k) * ZLIST_I(jjz);
        jjz++;
        jju++;
      }

    if (leg_j % 2 == 0) {
      int mb = leg_j / 2;
      for (int ma = 0; ma < mb; ma++) {
        for (int k = 0; k < 3; k++) sum[k] += DULIST_R(jju,k) * ZLIST_R(jjz) + DULIST_I(jju,k) * ZLIST_I(jjz);
        jjz++;
        jju++;
      }
      for (int k = 0; k < 3; k++) sum[k] += static_cast<DbRealT>(0.5) * ( DULIST_R(jju,k) * ZLIST_R(jjz) + DULIST_I(jju,k) * ZLIST_I(jjz) );
    }
  }

  template<class ZiRealT, class UiRealT, class DbRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_dbidrj_mono( // READ ONLY
                                               int twojmax, int idxb_max
                                             , int const * __restrict__ idxz_block
                                             , SnapInternal::SNA_BINDICES const * __restrict__ idxb
                                             , bool bnorm_flag
                                             , ZiRealT const * __restrict__ zlist_r
                                             , ZiRealT const * __restrict__ zlist_i
                                             , UiRealT const * __restrict__ dulist_r
                                             , UiRealT const * __restrict__ dulist_i
                                               // WRITE ONLY
                                             , DbRealT * __restrict__ dbdr ) // idxb_max*3, [jjb*3+xyz]
  {
    for (int jjb = 0; jjb < idxb_max; jjb++)
    {
      const int j1 = IDXB(jjb).j1;
      const int j2 = IDXB(jjb).j2;
      const int j  = IDXB(jjb).j;

      DbRealT sum[3] = { static_cast<DbRealT>(0), static_cast<DbRealT>(0), static_cast<DbRealT>(0) };
      DbRealT leg[3];

      leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
      snap_dbidrj_leg( j, IDXZ_BLOCK(j1,j2,j), IDXU_BLOCK(j), zlist_r, zlist_i, dulist_r, dulist_i, leg );
      for (int k = 0; k < 3; k++) sum[k] += leg[k];

      const DbRealT w1 = bnorm_flag ? static_cast<DbRealT>(1) : static_cast<DbRealT>(j+1) / static_cast<DbRealT>(j1+1);
      leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
      snap_dbidrj_leg( j1, IDXZ_BLOCK(j,j2,j1), IDXU_BLOCK(j1), zlist_r, zlist_i, dulist_r, dulist_i, leg );
      for (int k = 0; k < 3; k++) sum[k] += w1 * leg[k];

      const DbRealT w2 = bnorm_flag ? static_cast<DbRealT>(1) : static_cast<DbRealT>(j+1) / static_cast<DbRealT>(j2+1);
      leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
      snap_dbidrj_leg( j2, IDXZ_BLOCK(j,j1,j2), IDXU_BLOCK(j2), zlist_r, zlist_i, dulist_r, dulist_i, leg );
      for (int k = 0; k < 3; k++) sum[k] += w2 * leg[k];

      for (int k = 0; k < 3; k++) dbdr[jjb*3+k] = static_cast<DbRealT>(2) * sum[k];
    }
  }

  // Per-neighbor wrapper: builds this neighbor's raw U (needed by compute_duidrj),
  // its dU/dr, then the bispectrum derivative w.r.t. this neighbor's position.
  // ext must expose m_U_array and m_DU_array (see SnapBSExtStorage).
  template<class RijRealT, class RootPQRealT, class ZiRealT, class DbRealT, class ExtT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_neighbor_dbidrj(
                       int twojmax, int idxu_max, int idxb_max
                     , RijRealT wj, RijRealT rcut, RijRealT sinnerij, RijRealT dinnerij
                     , RijRealT x, RijRealT y, RijRealT z, RijRealT z0, RijRealT r
                     , RootPQRealT const * __restrict__ rootpqarray
                     , int const * __restrict__ idxz_block
                     , SnapInternal::SNA_BINDICES const * __restrict__ idxb
                     , ZiRealT const * __restrict__ zlist_r
                     , ZiRealT const * __restrict__ zlist_i
                     , RijRealT rmin0, RijRealT rfac0, bool switch_flag, bool switch_inner_flag, bool bnorm_flag
                       // WRITE ONLY
                     , DbRealT * __restrict__ dbdr
                     , ExtT & ext )
  {
    snap_compute_uarray( twojmax, rootpqarray, ext.m_U_array.r(), ext.m_U_array.i(), x, y, z, z0, r );
    snap_compute_duidrj( twojmax, idxu_max, x, y, z, rcut, wj,
                          ext.m_U_array.r(), ext.m_U_array.i(), rootpqarray,
                          sinnerij, dinnerij, rmin0, rfac0, switch_flag, switch_inner_flag, false /* chem_flag: mono-element only */,
                          ext.m_DU_array.r(), ext.m_DU_array.i() );
    snap_compute_dbidrj_mono( twojmax, idxb_max, idxz_block, idxb, bnorm_flag,
                               zlist_r, zlist_i, ext.m_DU_array.r(), ext.m_DU_array.i(), dbdr );
  }

  // Multi-element (chem_flag==true) counterpart to snap_compute_dbidrj_mono, ported literally from
  // LAMMPS ML-SNAP's SNA::compute_dbidrj() chem_flag branch (src/ML-SNAP/sna.cpp) -- not
  // "generalized from memory", the 3-leg idouble/itriple index algebra below (in particular leg 2/3's
  // swapped elem1<->elem3 triple index) is copy-verified against that source. zlist is the full
  // nelements^2*idxz_max-sized multi-element Z array (idouble = elem1*nelements+elem2 selects a
  // idxz_max-sized block, exactly snap_compute_zi's own output layout); dulist is this ONE neighbor's
  // ordinary (non-widened) per-pair dU/dr, unaffected by chem_flag (LAMMPS's own compute_duidrj is
  // chem_flag-agnostic too, see snap_compute_duidrj.h). elem3 is this neighbor's own element (fixed
  // for the whole call, the caller's jelem), matching LAMMPS's own `elem3 = elem_duarray`. Output
  // dbdr is nelements^3-widened: dbdr[(itriple*idxb_max+jjb)*3+xyz], itriple=(elem1*nelements+elem2)
  // *nelements+elem3 -- caller zero-inits its own idxb_max*nelements^3*3-sized buffer (see zero loop
  // below, matches LAMMPS zeroing its own dblist once at the top before this elem-restricted fill).
  template<class ZiRealT, class UiRealT, class DbRealT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_dbidrj_multi( // READ ONLY
                                               int twojmax, int idxb_max, int idxz_max, int nelements, int elem3
                                             , int const * __restrict__ idxz_block
                                             , SnapInternal::SNA_BINDICES const * __restrict__ idxb
                                             , bool bnorm_flag
                                             , ZiRealT const * __restrict__ zlist_r
                                             , ZiRealT const * __restrict__ zlist_i
                                             , UiRealT const * __restrict__ dulist_r
                                             , UiRealT const * __restrict__ dulist_i
                                               // WRITE ONLY
                                             , DbRealT * __restrict__ dbdr ) // idxb_max*nelements^3*3, [(itriple*idxb_max+jjb)*3+xyz]
  {
    const int ntriples = nelements * nelements * nelements;
    for (int i = 0; i < idxb_max*ntriples*3; i++) dbdr[i] = static_cast<DbRealT>(0);

    for (int jjb = 0; jjb < idxb_max; jjb++)
    {
      const int j1 = IDXB(jjb).j1;
      const int j2 = IDXB(jjb).j2;
      const int j  = IDXB(jjb).j;

      for (int elem1 = 0; elem1 < nelements; elem1++)
      for (int elem2 = 0; elem2 < nelements; elem2++)
      {
        DbRealT leg[3];

        // Leg 1 (weight 1): dudr(j,...) . z(j1,j2,j), itriple=(elem1,elem2,elem3)
        {
          const int idouble = elem1*nelements + elem2;
          const int itriple = idouble*nelements + elem3;
          leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
          snap_dbidrj_leg( j, IDXZ_BLOCK(j1,j2,j), IDXU_BLOCK(j),
                            zlist_r + static_cast<size_t>(idouble)*idxz_max, zlist_i + static_cast<size_t>(idouble)*idxz_max,
                            dulist_r, dulist_i, leg );
          DbRealT * const out = dbdr + (static_cast<size_t>(itriple)*idxb_max+jjb)*3;
          for (int k=0; k<3; k++) out[k] += static_cast<DbRealT>(2) * leg[k];
        }

        // Leg 2 (weight bnorm_flag?1:(j+1)/(j1+1)): dudr(j1,...) . z(j,j2,j1), itriple=(elem3,elem2,elem1)
        {
          const int idouble = elem1*nelements + elem2;
          const int itriple = (elem3*nelements + elem2)*nelements + elem1;
          const DbRealT w1 = bnorm_flag ? static_cast<DbRealT>(1) : static_cast<DbRealT>(j+1) / static_cast<DbRealT>(j1+1);
          leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
          snap_dbidrj_leg( j1, IDXZ_BLOCK(j,j2,j1), IDXU_BLOCK(j1),
                            zlist_r + static_cast<size_t>(idouble)*idxz_max, zlist_i + static_cast<size_t>(idouble)*idxz_max,
                            dulist_r, dulist_i, leg );
          DbRealT * const out = dbdr + (static_cast<size_t>(itriple)*idxb_max+jjb)*3;
          for (int k=0; k<3; k++) out[k] += static_cast<DbRealT>(2) * w1 * leg[k];
        }

        // Leg 3 (weight bnorm_flag?1:(j+1)/(j2+1)): dudr(j2,...) . z(j,j1,j2), itriple=(elem1,elem3,elem2)
        {
          const int idouble = elem2*nelements + elem1;
          const int itriple = (elem1*nelements + elem3)*nelements + elem2;
          const DbRealT w2 = bnorm_flag ? static_cast<DbRealT>(1) : static_cast<DbRealT>(j+1) / static_cast<DbRealT>(j2+1);
          leg[0] = leg[1] = leg[2] = static_cast<DbRealT>(0);
          snap_dbidrj_leg( j2, IDXZ_BLOCK(j,j1,j2), IDXU_BLOCK(j2),
                            zlist_r + static_cast<size_t>(idouble)*idxz_max, zlist_i + static_cast<size_t>(idouble)*idxz_max,
                            dulist_r, dulist_i, leg );
          DbRealT * const out = dbdr + (static_cast<size_t>(itriple)*idxb_max+jjb)*3;
          for (int k=0; k<3; k++) out[k] += static_cast<DbRealT>(2) * w2 * leg[k];
        }
      }
    }
  }

  // Multi-element per-neighbor wrapper, sibling to snap_compute_neighbor_dbidrj. elem3 = this
  // neighbor's own element (caller's jelem, matching the same value snap_add_nbh_contrib_to_uarraytot
  // used to place this neighbor's U contribution in the value-only pass).
  template<class RijRealT, class RootPQRealT, class ZiRealT, class DbRealT, class ExtT>
  ONIKA_HOST_DEVICE_FUNC
  static inline void snap_compute_neighbor_dbidrj_multi(
                       int twojmax, int idxu_max, int idxb_max, int idxz_max, int nelements, int elem3
                     , RijRealT wj, RijRealT rcut, RijRealT sinnerij, RijRealT dinnerij
                     , RijRealT x, RijRealT y, RijRealT z, RijRealT z0, RijRealT r
                     , RootPQRealT const * __restrict__ rootpqarray
                     , int const * __restrict__ idxz_block
                     , SnapInternal::SNA_BINDICES const * __restrict__ idxb
                     , ZiRealT const * __restrict__ zlist_r
                     , ZiRealT const * __restrict__ zlist_i
                     , RijRealT rmin0, RijRealT rfac0, bool switch_flag, bool switch_inner_flag, bool bnorm_flag
                       // WRITE ONLY
                     , DbRealT * __restrict__ dbdr
                     , ExtT & ext )
  {
    snap_compute_uarray( twojmax, rootpqarray, ext.m_U_array.r(), ext.m_U_array.i(), x, y, z, z0, r );
    snap_compute_duidrj( twojmax, idxu_max, x, y, z, rcut, wj,
                          ext.m_U_array.r(), ext.m_U_array.i(), rootpqarray,
                          sinnerij, dinnerij, rmin0, rfac0, switch_flag, switch_inner_flag, true /* chem_flag */,
                          ext.m_DU_array.r(), ext.m_DU_array.i() );
    snap_compute_dbidrj_multi( twojmax, idxb_max, idxz_max, nelements, elem3, idxz_block, idxb, bnorm_flag,
                                zlist_r, zlist_i, ext.m_DU_array.r(), ext.m_DU_array.i(), dbdr );
  }

}
