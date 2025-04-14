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

#include <md/snap/snap_force_op.h>

namespace md
{
  using namespace exanb;

  // Force operator
  template<class SnapConfParamT, class ComputeBufferT, class CellParticlesT>
  struct SnapXSForceOp<SnapConfParamT,ComputeBufferT,CellParticlesT,true>
  {
    const SnapConfParamT snaconf;
    
    const size_t * const __restrict__ cell_particle_offset = nullptr;
    const double * const __restrict__ beta = nullptr;
    const double * const __restrict__ bispectrum = nullptr;
    
    const double * const __restrict__ coeffelem = nullptr;
    const unsigned int coeffelem_size = 0;
    const unsigned int ncoeff = 0;
   
    const double * const __restrict__ wjelem = nullptr; // data of m_factor in snap_ctx
    const double * const __restrict__ radelem = nullptr;
    const double * const __restrict__ sinnerelem = nullptr;
    const double * const __restrict__ dinnerelem = nullptr;
    const double rcutfac = 0.0;
    const bool eflag = false;
    const bool quadraticflag = false;
    
    // md conversion specific falgs
    const bool conv_energy_units = true;
    const double conv_energy_factor = ONIKA_CONST_QUANTITY( 1. * eV ).convert();

    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& en,
      double& fx,
      double& fy,
      double& fz,
      int type,
      CellParticlesT cells
      ) const
    {
      FakeMat3d virial;
      ComputePairOptionalLocks<false> locks;
      FakeParticleLock lock_a;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& fx,
      double& fy,
      double& fz,
      int type,
      CellParticlesT cells
      ) const
    {
      FakeMat3d virial;
      double en = 0.0;
      ComputePairOptionalLocks<false> locks;
      FakeParticleLock lock_a;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& en,
      double& fx,
      double& fy,
      double& fz,
      int type,
      Mat3d& virial,
      CellParticlesT cells
      ) const
    {
      ComputePairOptionalLocks<false> locks;
      FakeParticleLock lock_a;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& fx,
      double& fy,
      double& fz,
      int type,
      Mat3d& virial,
      CellParticlesT cells
      ) const
    {
      double en = 0.0;
      ComputePairOptionalLocks<false> locks;
      FakeParticleLock lock_a;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    template<class GridCellLocksT, class ParticleLockT>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& en,
      double& fx,
      double& fy,
      double& fz,
      int type,
      CellParticlesT cells,
      GridCellLocksT locks,
      ParticleLockT& lock_a
      ) const
    {
      FakeMat3d virial;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    template<class GridCellLocksT, class ParticleLockT>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      size_t n,
      ComputeBufferT& buf,
      double& fx,
      double& fy,
      double& fz,
      int type,
      CellParticlesT cells,
      GridCellLocksT locks,
      ParticleLockT& lock_a
      ) const
    {
      FakeMat3d virial;
      double en = 0.0;
      this->operator () ( n,buf,en,fx,fy,fz,type,virial,cells,locks,lock_a);
    }

    template<class Mat3dT,class GridCellLocksT, class ParticleLockT>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator ()
      (
      int jnum ,
      ComputeBufferT& buf,
      double& en,
      double& fx,
      double& fy,
      double& fz,
      int type,
      Mat3dT& virial ,
      CellParticlesT cells,
      GridCellLocksT locks,
      ParticleLockT& lock_a
      ) const
    {
      static constexpr bool compute_virial = std::is_same_v< Mat3dT , Mat3d >;
      static constexpr bool CPAA = gpu_device_execution();
      static constexpr bool LOCK = ! gpu_device_execution();

      assert( ncoeff == static_cast<unsigned int>(snaconf.ncoeff) );

      // energy and force contributions to the particle
      Mat3dT _vir; // default constructor defines all elements to 0
      double _fx = 0.;
      double _fy = 0.;
      double _fz = 0.;

      // start of SNA
      const int itype = type;
      const double radi = radelem[itype];
      ONIKA_CU_BLOCK_SHARED int ninside;
      ONIKA_CU_BLOCK_SHARED uint8_t inside_idx[buf.MaxNeighbors];
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        ninside = 0;
        buf.ext.init( snaconf );
      }
      ONIKA_CU_BLOCK_SYNC();
      
      //for (int jj = 0; jj < jnum; jj++)
      ONIKA_CU_BLOCK_SIMD_FOR(int,jj,0,jnum)
      {
        const double rsq = buf.d2[jj];
        const int jtype = buf.nbh_pt[jj][field::type];
        const double cut_ij = ( radi + radelem[jtype] ) * rcutfac;
        const double cutsq_ij = cut_ij * cut_ij;
        const bool is_nbh_valid = ( rsq < cutsq_ij && rsq > 1e-20 );
        const int tl_ninside = ONIKA_CU_ATOMIC_ADD( ninside , int(is_nbh_valid) );
        if( is_nbh_valid ) inside_idx[tl_ninside] = jj;
      }
//      ONIKA_CU_BLOCK_SYNC();

      const double* __restrict__ betaloc = coeffelem + itype * (snaconf.ncoeff + 1 ) + 1;
      //const int idxu_max = snaconf.idxu_max; // used by macro ULIST_J_A
      
      snap_uarraytot_zero( snaconf.nelements, snaconf.idxu_max, buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i() , ONIKA_CU_THREAD_IDX, ONIKA_CU_BLOCK_SIZE );
      snap_uarraytot_init_wself( snaconf.nelements, snaconf.twojmax, snaconf.idxu_max, snaconf.wself, snaconf.wselfall_flag, buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i(), snaconf.chem_flag ? itype : 0 , ONIKA_CU_THREAD_IDX, ONIKA_CU_BLOCK_SIZE );
      ONIKA_CU_BLOCK_SYNC();

      //for (int jj = 0; jj < ninside; jj++)
      ONIKA_CU_BLOCK_SIMD_FOR(int,jj_in,0,ninside)
      {
        const int jj = inside_idx[jj_in];
        
        const int jtype = buf.nbh_pt[jj][field::type];
        const int jelem = snaconf.chem_flag ? jtype : 0 ;

        const double x = buf.drx[jj];
        const double y = buf.dry[jj];
        const double z = buf.drz[jj];
        const double rsq = buf.d2[jj];
        const double r = sqrt(rsq);
        const double rcutij_jj = ( radi + radelem[jtype] ) * rcutfac;
        const double theta0 = (r - snaconf.rmin0) * snaconf.rfac0 * M_PI / (rcutij_jj - snaconf.rmin0);
        const double z0 = r / tan(theta0);

        double sinnerij_jj = 0.0;
        double dinnerij_jj = 0.0;
        if (snaconf.switch_inner_flag) {
          sinnerij_jj = 0.5*(sinnerelem[itype]+sinnerelem[jtype]);
          dinnerij_jj = 0.5*(dinnerelem[itype]+dinnerelem[jtype]);
        }
        const double wj_jj = wjelem[jtype];
        const double sfac_jj = snap_compute_sfac( snaconf.rmin0, snaconf.switch_flag, snaconf.switch_inner_flag, r, rcutij_jj, sinnerij_jj, dinnerij_jj );

        snap_add_nbh_contrib_to_uarraytot( snaconf.twojmax, sfac_jj*wj_jj, x,y,z,z0,r, snaconf.rootpqarray, buf.ext.m_UTot_array.r() + snaconf.idxu_max * jelem, buf.ext.m_UTot_array.i() + snaconf.idxu_max * jelem, buf.ext , AtomicAccumFunctor{} );
      }
      ONIKA_CU_BLOCK_SYNC();
      /****************** end of Ui sum computation **********************/


      //snap_zero_yi_array( snaconf.nelements, snaconf.idxu_max_alt, buf.ext.m_Y_array.r(), buf.ext.m_Y_array.i() );
      {
        auto N = snaconf.idxu_max_alt * snaconf.nelements;
        double * __restrict__ ylist_r = buf.ext.m_Y_array.r();
        double * __restrict__ ylist_i = buf.ext.m_Y_array.i();
        //for(int i=0;i<N;++i)
        ONIKA_CU_BLOCK_SIMD_FOR(int,i,0,N)
        {
          YLIST_R(i) = 0.0;
          YLIST_I(i) = 0.0;
        }
      }
      ONIKA_CU_BLOCK_SYNC();

      snap_add_yi_contribution_alt( snaconf.nelements, snaconf.twojmax, snaconf.idxu_max, snaconf.idxz_max_alt
                              , snaconf.idxz_alt, snaconf.idxcg_block, snaconf.cglist
                              , snaconf.y_jju_map, snaconf.idxu_max_alt
                              , buf.ext.m_UTot_array.r(), buf.ext.m_UTot_array.i()
                              , snaconf.idxb_max, snaconf.idxb_block, snaconf.bnorm_flag
                              , buf.ext.m_Y_array.r(), buf.ext.m_Y_array.i()
                              , betaloc 
                              , ONIKA_CU_THREAD_IDX , ONIKA_CU_BLOCK_SIZE , AtomicAccumFunctor{}
                              );
      ONIKA_CU_BLOCK_SYNC();
      /******************* end of Yi computation ********************/


      //for (int jj_in = 0; jj_in < ninside; jj_in++) if( ONIKA_CU_THREAD_IDX == 0 )
      ONIKA_CU_BLOCK_SIMD_FOR(int,jj_in,0,ninside)
      {
        const int jj = inside_idx[jj_in];

        const int jtype = buf.nbh_pt[jj][field::type];
        const int jelem = snaconf.chem_flag ? jtype : 0 ;

        const double x = buf.drx[jj];
        const double y = buf.dry[jj];
        const double z = buf.drz[jj];
        const double rsq = buf.d2[jj];
        const double r = sqrt(rsq);
        const double rcutij_jj = ( radi + radelem[jtype] ) * rcutfac;
        const double theta0 = (r - snaconf.rmin0) * snaconf.rfac0 * M_PI / (rcutij_jj - snaconf.rmin0);
        const double z0 = r / tan(theta0);

        double sinnerij_jj = 0.0;
        double dinnerij_jj = 0.0;
        if (snaconf.switch_inner_flag) {
          sinnerij_jj = 0.5*(sinnerelem[itype]+sinnerelem[jtype]);
          dinnerij_jj = 0.5*(dinnerelem[itype]+dinnerelem[jtype]);
        }
        const double wj_jj = wjelem[jtype];

        double fij[3];
	      fij[0]=0.;
	      fij[1]=0.;
	      fij[2]=0.;

        add_nbh_contrib_to_force( snaconf.twojmax, snaconf.idxu_max, jelem , wj_jj, rcutij_jj, sinnerij_jj, dinnerij_jj , x, y, z, z0, r, rsq
                                , snaconf.rootpqarray, snaconf.y_jju_map, snaconf.idxu_max_alt
                                , snaconf.rmin0, snaconf.rfac0, snaconf.switch_flag, snaconf.switch_inner_flag, snaconf.chem_flag
                                , fij, buf.ext );

        fij[0] *= conv_energy_factor;
        fij[1] *= conv_energy_factor;
        fij[2] *= conv_energy_factor;

        if constexpr ( compute_virial )
        {
          const Mat3d v_contrib = tensor( Vec3d{ fij[0] , fij[1] , fij[2] }, Vec3d{ buf.drx[jj],buf.dry[jj],buf.drz[jj] } );
          _vir += v_contrib * -1.0;
        }

        _fx += fij[0];
        _fy += fij[1];
        _fz += fij[2];
        
        size_t cell_b=0, p_b=0;
        buf.nbh.get(jj, cell_b, p_b);
        concurent_add_contributions<ParticleLockT,CPAA,LOCK,double,double,double> (
            locks[cell_b][p_b]
          , cells[cell_b][field::fx][p_b], cells[cell_b][field::fy][p_b], cells[cell_b][field::fz][p_b]
          , -fij[0]                      , -fij[1]                      , -fij[2] );
      }
      // ONIKA_CU_BLOCK_SYNC();

      if constexpr ( compute_virial )
        concurent_add_contributions<ParticleLockT,CPAA,LOCK,double,double,double,Mat3d> ( lock_a, fx, fy, fz, virial, _fx, _fy, _fz, _vir );
      else
        concurent_add_contributions<ParticleLockT,CPAA,LOCK,double,double,double> ( lock_a, fx, fy, fz, _fx, _fy, _fz );
      
      // ONIKA_CU_BLOCK_SYNC();

      if constexpr ( SnapConfParamT::HasEneryField ) if (eflag)
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          double _en = 0.;
          const long bispectrum_ii_offset = snaconf.ncoeff * ( cell_particle_offset[buf.cell] + buf.part );
          const double * const coeffi = coeffelem /*[itype]*/;
	        double evdwl = coeffi[itype * (snaconf.ncoeff + 1)];

          for (int icoeff = 0; icoeff < snaconf.ncoeff; icoeff++)
          {
            evdwl += betaloc[icoeff] * bispectrum[ bispectrum_ii_offset + icoeff ] /*bispectrum[ii][icoeff]*/ ;
          }
          // quadratic contributions
          if (quadraticflag)
          {
            int k = snaconf.ncoeff+1;
            for (int icoeff = 0; icoeff < snaconf.ncoeff; icoeff++) 
            {
              double bveci = bispectrum[ bispectrum_ii_offset + icoeff ] /*bispectrum[ii][icoeff]*/ ;
              evdwl += 0.5*coeffi[k++]*bveci*bveci;
              for (int jcoeff = icoeff+1; jcoeff < snaconf.ncoeff; jcoeff++) 
              {
                double bvecj = bispectrum[ bispectrum_ii_offset + jcoeff ] /*bispectrum[ii][jcoeff]*/ ;
                evdwl += coeffi[k++]*bveci*bvecj;
              }
            }
          }
          evdwl *= 1.;//scale[itype][itype];
          _en = evdwl; // ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);
          if( conv_energy_units )
          {
            _en *= conv_energy_factor;
          }
          
          en += _en;
        }
      }
      
      // ONIKA_CU_BLOCK_SYNC();
    }
  };

}

