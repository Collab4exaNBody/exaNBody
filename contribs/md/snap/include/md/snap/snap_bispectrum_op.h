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

#include <md/snap/snap_compute_ui.h>
#include <md/snap/snap_compute_zi.h>
#include <md/snap/snap_compute_bi.h>

namespace md
{

  using namespace exanb;

  // Bispectrum evaluation operator
  template<class RealT, class RijRealT, class SnapConfParamT>
  struct BispectrumOpRealT
  {
    const SnapConfParamT snaconf;
        
    const size_t * const __restrict__ cell_particle_offset = nullptr;
    const RealT * const __restrict__ beta = nullptr;
    RealT * const __restrict__ bispectrum = nullptr;
    
    const RealT * const __restrict__ coeffelem = nullptr;
    const long ncoeff = 0;
    
    const RealT * const __restrict__ wjelem = nullptr; // data of m_factor in snap_ctx
    const RealT * const __restrict__ radelem = nullptr;
    const RealT * const __restrict__ sinnerelem = nullptr;
    const RealT * const __restrict__ dinnerelem = nullptr;
    const RijRealT rcutfac = 0.0;
    const bool eflag = false;
    const bool quadraticflag = false;

    template<class ComputeBufferT, class CellParticlesT>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator () ( int jnum, ComputeBufferT& buf, int itype, CellParticlesT cells) const
    {
      assert( ncoeff == static_cast<unsigned int>(snaconf.ncoeff) );
      
      buf.ext.init( snaconf );

      // start of SNA
      const RijRealT radi = radelem[itype];

      int ninside = 0;
      for (int jj = 0; jj < jnum; jj++)
      {
//        const double delx = buf.drx[jj];
//        const double dely = buf.dry[jj];
//        const double delz = buf.drz[jj];
        const RijRealT rsq = buf.d2[jj];
	      const int jtype = buf.nbh_pt[jj][field::type];
	      const RijRealT cut_ij = ( radi + radelem[jtype] ) * rcutfac;
	      const RijRealT cutsq_ij = cut_ij * cut_ij;
        if( rsq < cutsq_ij && rsq > static_cast<RijRealT>(1e-20) )
        {
          if( ninside != jj ) { buf.copy( jj , ninside ); }
          ninside++;
        }
      }
/*
      snap_compute_ui( snaconf.nelements, snaconf.twojmax, snaconf.idxu_max 
                     , snabuf.element, buf.drx,buf.dry,buf.drz , snabuf.rcutij, snaconf.rootpqarray, snabuf.sinnerij, snabuf.dinnerij, snabuf.wj
                     , snaconf.wselfall_flag, snaconf.switch_flag, snaconf.switch_inner_flag, snaconf.chem_flag
                     , snaconf.wself, snaconf.rmin0, snaconf.rfac0
                     , snabuf.ulist_r_ij, snabuf.ulist_i_ij, snabuf.ulisttot_r, snabuf.ulisttot_i
                     , ninside, snaconf.chem_flag ? itype : 0);
*/
      // unrolled here after to avoid writing results to all ulist_r for each neighbors (we never reuse them afterward)
      /************ being of UiTot computation ******************/

//      double UiTot_array_r[ snaconf.idxu_max * snaconf.nelements ];
//      double UiTot_array_i[ snaconf.idxu_max * snaconf.nelements ];
      snap_uarraytot_zero( snaconf.nelements, snaconf.idxu_max, buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i() );
      snap_uarraytot_init_wself( snaconf.nelements, snaconf.twojmax, snaconf.idxu_max, snaconf.wself, snaconf.wselfall_flag, buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i(), snaconf.chem_flag ? itype : 0 );
      for (int jj = 0; jj < ninside; jj++)
      {
        const int jtype = buf.nbh_pt[jj][field::type];
        const int jelem = snaconf.chem_flag ? jtype : 0 ;

        const RijRealT x = buf.drx[jj];
        const RijRealT y = buf.dry[jj];
        const RijRealT z = buf.drz[jj];
        const RijRealT rsq = buf.d2[jj];
        const RijRealT r = sqrt(rsq);
        const RijRealT rcutij_jj = ( radi + radelem[jtype] ) * rcutfac;
        const RijRealT theta0 = (r - snaconf.rmin0) * snaconf.rfac0 * M_PI / (rcutij_jj - snaconf.rmin0);
        const RijRealT z0 = r / tan(theta0);

        //double Ui_array_r[snaconf.idxu_max];
        //double Ui_array_i[snaconf.idxu_max];        
        //snap_compute_uarray( snaconf.twojmax, snaconf.rootpqarray, Ui_array_r, Ui_array_i, x, y, z, z0, r );

        RijRealT sinnerij_jj = static_cast<RijRealT>(0.0);
        RijRealT dinnerij_jj = static_cast<RijRealT>(0.0);
        if( snaconf.switch_inner_flag )
        {
          sinnerij_jj = static_cast<RijRealT>(0.5)*(sinnerelem[itype]+sinnerelem[jtype]);
          dinnerij_jj = static_cast<RijRealT>(0.5)*(dinnerelem[itype]+dinnerelem[jtype]);
        }
        const RijRealT wj_jj = wjelem[jtype];
        const RijRealT sfac_jj = snap_compute_sfac( static_cast<RijRealT>(snaconf.rmin0), snaconf.switch_flag, snaconf.switch_inner_flag, r, rcutij_jj, sinnerij_jj, dinnerij_jj );

        //snap_add_uarraytot( snaconf.twojmax, jelem, snaconf.idxu_max, sfac_jj*wj_jj, Ui_array_r, Ui_array_i, UiTot_array_r /*snabuf.ulisttot_r*/, UiTot_array_i /*snabuf.ulisttot_i*/ );
        snap_add_nbh_contrib_to_uarraytot( snaconf.twojmax, sfac_jj*wj_jj, x,y,z,z0,r, snaconf.rootpqarray, buf.ext.m_UTot_array.r() + snaconf.idxu_max * jelem, buf.ext.m_UTot_array.i() + snaconf.idxu_max * jelem, buf.ext );
      }

      /****************** end of UiTot computation **********************/
//      double Zi_array_r[ snaconf.idxz_max * snaconf.ndoubles ];
//      double Zi_array_i[ snaconf.idxz_max * snaconf.ndoubles ];
      snap_compute_zi( snaconf.nelements, snaconf.idxz_max, snaconf.idxu_max, snaconf.twojmax
                     , snaconf.idxcg_block
                     , snaconf.idxz, snaconf.cglist, buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i()
                     , snaconf.bnorm_flag, buf.ext.m_Z_array.r(), buf.ext.m_Z_array.i() );

      const long bispectrum_ii_offset = snaconf.ncoeff * ( cell_particle_offset[buf.cell] + buf.part );
      snap_compute_bi( snaconf.nelements, snaconf.idxz_max, snaconf.idxb_max, snaconf.idxu_max, snaconf.twojmax
                     , snaconf.idxz_block
                     , snaconf.idxz, snaconf.idxb
                     , buf.ext.m_Z_array.r(), buf.ext.m_Z_array.i()
                     , buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i()
                     , snaconf.bzero , snaconf.bzero_flag, snaconf.wselfall_flag
                     , bispectrum + bispectrum_ii_offset //snabuf.blist
                     , snaconf.chem_flag ? itype : 0 );
    }
    
  };

//  template<class SnapConfParamT>
//  using BispectrumOp =  BispectrumOpRealT<double,SnapConfParamT>;
}

namespace exanb
{
  template<class RealT, class RijRealT, class SnapConfParamT>
  struct ComputePairTraits< md::BispectrumOpRealT<RealT,RijRealT,SnapConfParamT> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool ComputeBufferCompatible      = true;
    static inline constexpr bool BufferLessCompatible         = false;
    static inline constexpr bool CudaCompatible               = true;
  };

}
