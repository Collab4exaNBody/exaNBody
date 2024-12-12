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
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/compute/compute_cell_particle_pairs.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS constant
#include <exanb/particle_neighbors/chunk_neighbors.h> // for MAX_PARTICLE_NEIGHBORS constant

// this allows for parallel compilation of templated operator for each available field set
namespace microStamp
{
  using namespace exanb;

  // LennardJones Parameters
  struct LennardJonesParms
  {
    double epsilon = 0.0;
    double sigma = 0.0;
  };

  ONIKA_HOST_DEVICE_FUNC inline void lj_compute_energy(const LennardJonesParms& p, double r, double& e, double& de)
  {
    assert( r > 0. );
    double ratio = p.sigma / r;
    double ratio6 = pow(ratio,6);    // attractive
    double ratio12 = pow(ratio,12);  // repulsive
    e = 4. * p.epsilon * (ratio12-ratio6) ;
    de = ( -24. * p.epsilon * (2.*ratio12-ratio6) ) / r;
  }

  // interaction potential compute functor
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) LennardJonesForceFunctor
  {
    // potential function parameters
    const LennardJonesParms& m_params;

    // concrete type of computation buffer and particle container may vary,
    // we use templates here to adapat to various situations
    template<class ComputePairBufferT, class CellParticlesT> 
    inline void operator () (
      size_t n,                          // number of neighbor particles
      const ComputePairBufferT& buffer,  // neighbors buffer
      double& e,                        // central particle's potential energy reference
      double& fx,                        // central particle's force/X reference
      double& fy,                        // central particle's force/Y reference
      double& fz,                        // central particle's force/Z reference
      CellParticlesT* cells              // arrays of all cells, in case we need to chase for additional particle informations
      ) const
    {
      // local energy and force contributions to the particle
      double _e = 0.; 
      double _fx = 0.;
      double _fy = 0.;
      double _fz = 0.;

#     pragma omp simd reduction(+:_e,_fx,_fy,_fz)
      for(size_t i=0;i<n;i++)
      {
        const double r = std::sqrt(buffer.d2[i]);
        double pair_e=0.0, pair_de=0.0;
        lj_compute_energy( m_params, r, pair_e, pair_de );
        const auto interaction_weight = buffer.nbh_data.get(i);
        pair_e *= interaction_weight;
        pair_de *= interaction_weight / r;        
        _fx += pair_de * buffer.drx[i];  // force is energy derivative multiplied by rij vector, sum force contributions for all neighbor particles
        _fy += pair_de * buffer.dry[i];
        _fz += pair_de * buffer.drz[i];
        _e += .5 * pair_e;
      }
      e  += _e;   // add local contribution to central particle's attributes
      fx += _fx;
      fy += _fy;
      fz += _fz;
    }

    // ComputeBuffer less computation without virial
    template<class CellParticlesT>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (
        Vec3d dr
      , double d2
      , double& e
      , double& fx
      , double& fy
      , double& fz
      , CellParticlesT* cells
      , size_t neighbor_cell
      , size_t neighbor_particle
      , double interaction_weight ) const
    {
      const double r = sqrt(d2);
      double pair_e = 0.0 , pair_de = 0.0;
      lj_compute_energy( m_params, r, pair_e, pair_de );
      pair_e *= interaction_weight;
      pair_de *= interaction_weight / r;        
      fx += pair_de * dr.x;
      fy += pair_de * dr.y;
      fz += pair_de * dr.z;
      e += .5 * pair_e;
    }

  };

}

namespace exanb
{

  // specialize functor traits to allow Cuda execution space
  template<>
  struct ComputePairTraits< microStamp::LennardJonesForceFunctor >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool ComputeBufferCompatible = true;
    static inline constexpr bool BufferLessCompatible = true;
    static inline constexpr bool CudaCompatible = true;
  };

}

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML
{
  using microStamp::LennardJonesParms;
  using exanb::UnityConverterHelper;
  using exanb::Quantity;

  template<> struct convert<LennardJonesParms>
  {
    static bool decode(const Node& node, LennardJonesParms& v)
    {
      v = LennardJonesParms{};
      if( !node.IsMap() ) { return false; }
      v.epsilon = node["epsilon"].as<Quantity>().convert();
      v.sigma   = node["sigma"]  .as<Quantity>().convert();
      return true;
    }
  };

}

namespace microStamp
{
  using namespace exanb;

  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_ep ,field::_fx ,field::_fy ,field::_fz >
    >
  class LennardJonesForce : public OperatorNode
  {
    // ========= I/O slots =======================
    ADD_SLOT( LennardJonesParms , config , INPUT , REQUIRED , DocString{"Lennard-Jones potential parameters"} );
    ADD_SLOT( double , rcut , INPUT , 0.0 , DocString{"Cutoff distance"} );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors , INPUT , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
    ADD_SLOT( bool , ghost , INPUT , false , DocString{"Enables computation in ghost cells"});
    ADD_SLOT( Domain , domain , INPUT , REQUIRED , DocString{"Simulation domain"});

    ADD_SLOT( double , rcut_max , INPUT_OUTPUT , 0.0 , DocString{"Updated max rcut"});
    ADD_SLOT( GridT , grid , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );

    // shortcut to the Compute buffer used (and passed to functor) by compute_pair_singlemat
    using ComputeBuffer = ComputePairBuffer2<false,false>;
    using CellParticles = typename GridT::CellParticles;
    using ComputeFields = FieldSet< field::_ep ,field::_fx ,field::_fy ,field::_fz >;

  public:
    // Operator execution
    inline void execute () override final
    {
      assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

      *rcut_max = std::max( *rcut , *rcut_max );
      if( grid->number_of_cells() == 0 ) { return; }

      ComputePairOptionalLocks<false> cp_locks {};
      exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
      auto force_buf = make_compute_pair_buffer<ComputeBuffer>();
      LennardJonesForceFunctor force_op = { *config };

      LinearXForm cp_xform { domain->xform() };
      auto optional = make_compute_pair_optional_args( nbh_it, ComputePairNullWeightIterator{} , cp_xform, cp_locks );
      compute_cell_particle_pairs( *grid, *rcut, *ghost, optional, force_buf, force_op , ComputeFields{} , DefaultPositionFields{} , parallel_execution_context() );      
    }

  };

  template<class GridT> using LennardJonesForceTmpl = LennardJonesForce<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {  
    OperatorNodeFactory::instance()->register_factory( "lennard_jones_force" , make_grid_variant_operator< LennardJonesForceTmpl > );
  }

}


