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

#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <exanb/compute/compute_cell_particle_pairs.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <exanb/core/particle_type_id.h>

#include <onika/file_utils.h>

#include <md/snap/snap_params.h>
#include <md/snap/snap_read_lammps.h>
#include <md/snap/snap_config.h>
#include <md/snap/snap_check_bispectrum.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>

#include <vector>
#include <memory>
#include <iostream>

#include <mpi.h>

#include <md/snap/snap_context.h>
#include <md/snap/snap_force_op.h>
#include <md/snap/snap_bispectrum_op.h>

namespace md
{

  using namespace exanb;
  using onika::memory::DEFAULT_ALIGNMENT;
//  using namespace SnapExt;

  struct ResetSnapCPBuf
  {
    template<class CPBufT>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (CPBufT & buf) const
    {
      buf.ext.reset();
    }
  };

  template<
      class GridT
    , class SnapRealT = double
    , class EpFieldT = unused_field_id_t 
    , class VirialFieldT = unused_field_id_t 
    , class = AssertGridHasFields< GridT, field::_fx ,field::_fy ,field::_fz >
    >
  class SnapForceRealT : public OperatorNode
  {
    // floating point precision configuration
    using ComputeBufferRealT = SnapRealT;
    using ConstantsRealT     = SnapRealT;
    using ForceRealT       = SnapRealT;
    using BSRealT  = SnapRealT;
    
    using SnapContext = SnapXSContextRealT<ConstantsRealT>;

    // ========= I/O slots =======================
    ADD_SLOT( MPI_Comm              , mpi               , INPUT , REQUIRED);
    ADD_SLOT( SnapParms             , parameters        , INPUT , REQUIRED );
    ADD_SLOT( double                , rcut_max          , INPUT_OUTPUT , 0.0 );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors   , INPUT , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
    ADD_SLOT( bool                  , ghost             , INPUT , false );
    ADD_SLOT( bool                  , conv_coef_units   , INPUT , false );
    ADD_SLOT( bool                  , trigger_thermo_state, INPUT , OPTIONAL );
    ADD_SLOT( GridT                 , grid              , INPUT_OUTPUT );
    ADD_SLOT( Domain                , domain            , INPUT , REQUIRED );
    ADD_SLOT( GridParticleLocks     , particle_locks    , INPUT , OPTIONAL , DocString{"particle spin locks"} );

    ADD_SLOT( ParticleTypeMap       , particle_type_map , INPUT , OPTIONAL ); // to reorder material indices if needed, to match indices used in snap parameters

    ADD_SLOT( long                  , timestep          , INPUT , REQUIRED , DocString{"Iteration number"} );
    ADD_SLOT( std::string           , bispectrumchkfile , INPUT , OPTIONAL , DocString{"file with reference values to check bispectrum correctness"} );
    ADD_SLOT( bool                  , scb_mode          , INPUT , false , DocString{"if true, enables block-wide collaborative computation of atom forces"} );
    
    ADD_SLOT( SnapContext           , snap_ctx          , PRIVATE );

    // shortcut to the Compute buffer used (and passed to functor) by compute_cell_particle_pairs
    static inline constexpr bool SnapUseWeights = false;
    static inline constexpr bool SnapUseNeighbors = true;

    template<class SnapConfParamT>
    using ComputeBuffer = ComputePairBuffer2< SnapUseWeights, SnapUseNeighbors
                                            , SnapXSForceExtStorage<SnapConfParamT,ComputeBufferRealT>, DefaultComputePairBufferAppendFunc
                                            , exanb::MAX_PARTICLE_NEIGHBORS, ComputePairBuffer2Weights
                                            , FieldSet<field::_type> >;

    template<class SnapConfParamT>
    using ComputeBufferBS = ComputePairBuffer2< SnapUseWeights, SnapUseNeighbors
                                            , SnapBSExtStorage<SnapConfParamT,ComputeBufferRealT>, DefaultComputePairBufferAppendFunc
                                            , exanb::MAX_PARTICLE_NEIGHBORS, ComputePairBuffer2Weights
                                            , FieldSet<field::_type> >;

    using CellParticles = typename GridT::CellParticles;

    // compile time constant indicating if grid has virial field
    static inline constexpr bool has_virial_field = GridHasField<GridT,VirialFieldT>::value;
    static inline constexpr bool has_energy_field = GridHasField<GridT,EpFieldT>::value;

    // template shortcuts
    template<int I> using ICST = onika::IntConst<I>;
    template<int jm> using ROParamsMonoElem = SnapInternal::ReadOnlySnapParametersRealT<ConstantsRealT,ICST<jm>,ICST<1>,has_energy_field>;

    // attributes processed during computation
    using ComputeFields = std::conditional_t< has_virial_field
                                            , std::conditional_t<  has_energy_field
                                                                , FieldSet< EpFieldT, field::_fx, field::_fy, field::_fz, field::_type, VirialFieldT >
                                                                , FieldSet< field::_fx, field::_fy, field::_fz, field::_type, VirialFieldT > > 
                                            , std::conditional_t< has_energy_field
                                                         , FieldSet< EpFieldT ,field::_fx ,field::_fy ,field::_fz ,field::_type >
                                                         , FieldSet< field::_fx ,field::_fy ,field::_fz ,field::_type > > 
                                            >;
    static constexpr ComputeFields compute_force_field_set{};
    static constexpr FieldSet< field::_type> compute_bispectrum_field_set{};
        
  public:
    
    // Operator execution
    inline void execute () override final
    {
      assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

      //ldbg << "rcut="<<snap_ctx->m_rcut <<std::endl << std::flush;
      if( snap_ctx->m_rcut == 0.0 )
      {
        std::string lammps_param = onika::data_file_path( parameters->lammps_param );
        std::string lammps_coef = onika::data_file_path( parameters->lammps_coef ); 
        ldbg << "Snap: read lammps files "<<lammps_param<<" and "<<lammps_coef<<std::endl << std::flush;
        SnapExt::snap_read_lammps(lammps_param, lammps_coef, snap_ctx->m_config, *conv_coef_units );
        ldbg <<"rfac0="<<snap_ctx->m_config.rfac0() <<", rmin0="<<snap_ctx->m_config.rmin0() <<", rcutfac="<<snap_ctx->m_config.rcutfac() 
             <<", twojmax="<<snap_ctx->m_config.twojmax()<<", bzeroflag="<<snap_ctx->m_config.bzeroflag()<<", nmat="<<snap_ctx->m_config.materials().size()
             <<", chemflag="<<snap_ctx->m_config.chemflag() <<std::endl;
        snap_ctx->m_rcut = snap_ctx->m_config.rcutfac(); // because LAMMPS uses angstrom while exastamp uses nm
      }

      *rcut_max = std::max( double(*rcut_max) , double(snap_ctx->m_rcut) );
      
      size_t n_cells = grid->number_of_cells();
      if( n_cells==0 )
      {
        return ;
      }

      if( ! particle_locks.has_value() )
      {
        fatal_error() << "No particle locks" << std::endl;
      }

      if( snap_ctx->m_coefs.empty() )
      {
        /*
        for( const auto& mat : snap_ctx->m_config.materials() )
        {
          ldbg << '\t' << mat.name() << ": radelem="<<mat.radelem()<<", weight="<<mat.weight()<<", ncoefs="<<mat.number_of_coefficients()<<std::endl;
          for(size_t i=0;i<mat.number_of_coefficients();i++)
          {
            ldbg << "\t\t" << mat.coefficient(i) << std::endl;
          }
        }
        */
        
        int nmat = snap_ctx->m_config.materials().size();
	      
        // temporay, enable mutiple species if they all have weight=1. modifications needed for true multimaterial
        snap_ctx->m_factor.assign( nmat, 1.0 );
        snap_ctx->m_radelem.assign( nmat, 0.0 );

        int cnt=0;
        for ( const auto& mat : snap_ctx->m_config.materials() )
        {
          snap_ctx->m_factor[cnt] = mat.weight();	    
          snap_ctx->m_radelem[cnt] = mat.radelem();
          cnt+=1;
        }

        size_t ncoefs_per_specy = snap_ctx->m_config.materials()[0].number_of_coefficients();
        snap_ctx->m_coefs.resize( nmat * ncoefs_per_specy );
        for(int j=0;j<nmat;j++)
        {
          const auto& mat = snap_ctx->m_config.materials()[j];
          for(size_t i=0;i<ncoefs_per_specy;i++)
          {
            snap_ctx->m_coefs[ j * ncoefs_per_specy + i ] = mat.coefficient(i);
          }
        }

      }

      if( snap_ctx->sna == nullptr )
      {
        snap_ctx->sna = new SnapInternal::SNARealT<ConstantsRealT>( new SnapInternal::Memory()
                                          , snap_ctx->m_config.rfac0() 
                                          , snap_ctx->m_config.twojmax() 
                                          , snap_ctx->m_config.rmin0()
                                          , snap_ctx->m_config.switchflag()
                                          , snap_ctx->m_config.bzeroflag()
                                          , snap_ctx->m_config.chemflag()
                                          , snap_ctx->m_config.bnormflag()
                                          , snap_ctx->m_config.wselfallflag()
                                          , snap_ctx->m_config.nelements()
                                          , snap_ctx->m_config.switchinnerflag()
                                          );
        snap_ctx->sna->init();        
      }
      ldbg << "*** Constant config allocation ***" << std::endl;
      snap_ctx->sna->memory->print( ldbg );

      ldbg << "Max number of neighbors = "<< chunk_neighbors->m_max_neighbors << std::endl;

      bool log_energy = false;
      if constexpr ( has_energy_field )
      {
        if( trigger_thermo_state.has_value() )
        {
          ldbg << "trigger_thermo_state = " << *trigger_thermo_state << std::endl;
          log_energy = *trigger_thermo_state ;
        }
        else
        {
          ldbg << "trigger_thermo_state missing " << std::endl;
        }
      }

//      const double cutsq = snap_ctx->m_rcut * snap_ctx->m_rcut;
      const bool eflag = log_energy || bispectrumchkfile.has_value();
      const bool quadraticflag = snap_ctx->m_config.quadraticflag();

      assert( snap_ctx->m_config.switchinnerflag() == snap_ctx->sna->switch_inner_flag );
      assert( snap_ctx->m_config.chemflag() == snap_ctx->sna->chem_flag );

      // exanb objects to perform computations with neighbors      
      ComputePairNullWeightIterator cp_weight{};
      exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
      LinearXForm cp_xform { domain->xform() };

      // constants to resize bispectrum and beta intermediate terms
      const size_t total_particles = grid->number_of_particles();
      size_t ncoefs_per_specy = snap_ctx->m_config.materials()[0].number_of_coefficients();
      int ncoeffall = ncoefs_per_specy; //_per_specysnap_ctx->m_coefs.size() ;
      int ncoeff = -1;
      
      if (!quadraticflag)
        ncoeff = ncoeffall - 1;
      else {
        ncoeff = sqrt(2*ncoeffall)-1;
        int ncoeffq = (ncoeff*(ncoeff+1))/2;
        int ntmp = 1+ncoeff+ncoeffq;
        if (ntmp != ncoeffall) {
          fatal_error() << "Incorrect SNAP coeff file" << std::endl;
        }
      }

      ldbg << "snap: quadratic="<<quadraticflag<<", eflag="<<eflag<<", ncoeff="<<ncoeff<<", ncoeffall="<<ncoeffall<<std::endl;

      auto snap_compute_specialized_snapconf = [&]( const auto & snapconf , auto c_use_coop_compute )
      {
        using SnapConfParamsT = std::remove_cv_t< std::remove_reference_t< decltype( snapconf ) > >;
        //snapconf.to_stream( ldbg );
      
        ComputePairOptionalLocks<true> cp_locks = { particle_locks->data() };
        auto optional = make_compute_pair_optional_args( nbh_it, cp_weight , cp_xform, cp_locks
                      , ComputePairTrivialCellFiltering{}, ComputePairTrivialParticleFiltering{}, grid->field_accessors_from_field_set(FieldSet<field::_type>{}) );

        if (quadraticflag || eflag)
        {
          // ************ compute_bispectrum(); ****************
          snap_ctx->m_bispectrum.clear();
          snap_ctx->m_bispectrum.resize( total_particles * ncoeff );

          BispectrumOpRealT<BSRealT,BSRealT,SnapConfParamsT> bispectrum_op {
                             snapconf,
                             grid->cell_particle_offset_data(), snap_ctx->m_beta.data(), snap_ctx->m_bispectrum.data(),
                             snap_ctx->m_coefs.data(), ncoeff,
                             snap_ctx->m_factor.data(), snap_ctx->m_radelem.data(),
                             nullptr, nullptr,
                             snap_ctx->m_rcut, eflag, quadraticflag };

          auto bs_buf = make_compute_pair_buffer< ComputeBufferBS<SnapConfParamsT> , ResetSnapCPBuf >();
          auto cp_fields = grid->field_accessors_from_field_set( compute_bispectrum_field_set );
          compute_cell_particle_pairs2( *grid, snap_ctx->m_rcut, *ghost, optional, bs_buf, bispectrum_op , cp_fields
                                      , DefaultPositionFields{}, parallel_execution_context() );

          // *********************************************        
          if( bispectrumchkfile.has_value() )
          {
            std::ostringstream oss; oss << *bispectrumchkfile << "." << *timestep;
            std::string file_name = onika::data_file_path( oss.str() );
            ldbg << "bispectrumchkfile is set, checking bispectrum from file "<< file_name << std::endl;
            snap_check_bispectrum(*mpi, *grid, file_name, ncoeff, snap_ctx->m_bispectrum.data() );
          }
        }
        
        std::true_type use_cells_accessor = {};
        using CellsAccessorT = std::remove_cv_t<std::remove_reference_t<decltype(grid->cells_accessor())> >;
        using CPBufT = ComputeBuffer<SnapConfParamsT>;
        ldbg << "max neighbors = " << CPBufT::MaxNeighbors << std::endl;
        SnapXSForceOpRealT<ForceRealT,ForceRealT,SnapConfParamsT,CPBufT,CellsAccessorT,c_use_coop_compute.value> force_op {
                           snapconf,
                           grid->cell_particle_offset_data(), snap_ctx->m_beta.data(), snap_ctx->m_bispectrum.data(),
                           snap_ctx->m_coefs.data(), static_cast<unsigned int>(snap_ctx->m_coefs.size()), static_cast<unsigned int>(ncoeff),
                           snap_ctx->m_factor.data(), snap_ctx->m_radelem.data(),
                           nullptr, nullptr,
                           snap_ctx->m_rcut, eflag, quadraticflag,
                           ! (*conv_coef_units) // if coefficients were not converted, then output energy/force must be converted
                           };
                           
        auto force_buf = make_compute_pair_buffer<CPBufT,ResetSnapCPBuf>();
        auto cp_fields = grid->field_accessors_from_field_set( compute_force_field_set );

        compute_cell_particle_pairs2( *grid, snap_ctx->m_rcut, *ghost, optional, force_buf, force_op , cp_fields
                                    , DefaultPositionFields{}, parallel_execution_context(), use_cells_accessor );
      };
      
      bool fallback_to_generic = false;
      const int JMax = snap_ctx->sna->twojmax / 2;
      static constexpr onika::BoolConst<true> use_shared_compute_buffer = {};
      static constexpr onika::BoolConst<false> no_shared_compute_buffer = {};

      if( snap_ctx->sna->nelements == 1 )
      {
        if( *scb_mode )
        {
               if( JMax == 3 ) snap_compute_specialized_snapconf( ROParamsMonoElem<3>(snap_ctx->sna) , use_shared_compute_buffer );
          else if( JMax == 4 ) snap_compute_specialized_snapconf( ROParamsMonoElem<4>(snap_ctx->sna) , use_shared_compute_buffer );
          else fallback_to_generic = true;
        }
        else
        {
               if( JMax == 3 ) snap_compute_specialized_snapconf( ROParamsMonoElem<3>(snap_ctx->sna) , no_shared_compute_buffer );
          else if( JMax == 4 ) snap_compute_specialized_snapconf( ROParamsMonoElem<4>(snap_ctx->sna) , no_shared_compute_buffer );
          else fallback_to_generic = true;
        }
      }
      else
      {
        fallback_to_generic = true;
      }
      
      if( fallback_to_generic )
      {
        snap_compute_specialized_snapconf( SnapInternal::ReadOnlySnapParametersRealT<ConstantsRealT,int,int,has_energy_field>( snap_ctx->sna ) , no_shared_compute_buffer );
      }
      
      ldbg << "Snap DONE (JMax="<<JMax<<",generic="<<std::boolalpha<<fallback_to_generic<<")"<<std::endl; 
    }

  };

  template<class GridT, class EpFieldT = unused_field_id_t , class VirialFieldT = unused_field_id_t>
  using SnapForceGenericFP64 = SnapForceRealT<GridT,double,EpFieldT,VirialFieldT>;

  template<class GridT, class EpFieldT = unused_field_id_t , class VirialFieldT = unused_field_id_t>
  using SnapForceGenericFP32 = SnapForceRealT<GridT,float,EpFieldT,VirialFieldT>;
}
