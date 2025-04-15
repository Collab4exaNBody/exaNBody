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

#include <onika/log.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/math/basic_types.h>
#include <onika/physics/units.h>

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS constant
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/compute/compute_cell_particle_pairs.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/particle_neighbors/chunk_neighbors.h> // for MAX_PARTICLE_NEIGHBORS constant

// this allows for parallel compilation of templated operator for each available field set
namespace exanb
{

  // AverageNeighbors Context
  template<class ValueT>
  struct AverageNeighborsExtStorage
  {
    ValueT m_sum = {};
    int m_nbh_count = 0.0;
    
    ONIKA_HOST_DEVICE_FUNC
    inline void reset() { m_sum = ValueT{}; m_nbh_count = 0; }

    ONIKA_HOST_DEVICE_FUNC
    inline auto avg() { return ( m_nbh_count > 0 ) ? ( m_sum / m_nbh_count ) : ( ValueT{} / 1 ); }
  };

  // interaction potential compute functor
  template<class AvgFieldT, class NbhFieldT>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) AverageNeighborsFunctor
  {
    // potential function parameters
    const double m_rcut_sq = {};
    AvgFieldT m_avg_field = {};
    NbhFieldT m_nbh_field = {};

    template<class ComputeBufferT, class CellParticlesT>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (ComputeBufferT& ctx, CellParticlesT cells, size_t cell_a , size_t p_a, exanb::ComputePairParticleContextStart ) const
    {
      ctx.ext.reset();
    }

    template<class ComputeBufferT, class CellParticlesT>
    ONIKA_HOST_DEVICE_FUNC ONIKA_ALWAYS_INLINE void operator () (ComputeBufferT& ctx, CellParticlesT cells, size_t cell_a, size_t p_a, exanb::ComputePairParticleContextStop ) const
    {
      cells[cell_a][m_avg_field][p_a] = ctx.ext.avg();
    }

    template<class ComputeBufferT, class CellParticlesT>
    ONIKA_HOST_DEVICE_FUNC ONIKA_ALWAYS_INLINE void operator () (
       ComputeBufferT& ctx
      , const Vec3d& dr, double d2 /* , double avg  */
      , CellParticlesT cells,size_t cell_b, size_t p_b
      , double /*scale*/) const
    {
      if( d2 <= m_rcut_sq )
      {
        ctx.ext.m_sum += cells[cell_b][m_nbh_field][p_b];
        ctx.ext.m_nbh_count ++;
      }
    }

  };

  // specialize functor traits to allow Cuda execution space
  template<class AvgFieldT, class NbhFieldT>
  struct ComputePairTraits< AverageNeighborsFunctor<AvgFieldT,NbhFieldT> >
  {
    static inline constexpr bool ComputeBufferCompatible = false;
    static inline constexpr bool BufferLessCompatible    = true;
    static inline constexpr bool CudaCompatible          = true;
    static inline constexpr bool HasParticleContextStart = true;    
    static inline constexpr bool HasParticleContext      = true;
    static inline constexpr bool HasParticleContextStop  = true;
  };

  template<class GridT, class AvgFieldT, class NbhFieldT>
  class AverageNeighbors : public OperatorNode
  {
    // ========= I/O slots =======================
    ADD_SLOT( double                    , rcut            , INPUT        , 0.0 , DocString{"Cutoff distance"} );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors , INPUT        , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
    ADD_SLOT( bool                      , ghost           , INPUT        , false , DocString{"Enables computation in ghost cells"});
    ADD_SLOT( Domain                    , domain          , INPUT        , REQUIRED , DocString{"Simulation domain"});
    ADD_SLOT( AvgFieldT                 , avg_field       , INPUT        , AvgFieldT{} , DocString{"name of resulting average field"});
    ADD_SLOT( NbhFieldT                 , nbh_field       , INPUT        , NbhFieldT{} , DocString{"name of averaged neighbor quantity"});
    ADD_SLOT( double                    , rcut_max        , INPUT_OUTPUT , 0.0 , DocString{"Updated max rcut"});
    ADD_SLOT( GridT                     , grid            , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );

    // shortcut to the Compute buffer used (and passed to functor) by compute_pair_singlemat
    using ComputeBuffer = ComputePairBuffer2<false,false,AverageNeighborsExtStorage<typename AvgFieldT::value_type> >;

  public:
    // Operator execution
    inline void execute () override final
    {
      assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

      *rcut_max = std::max( *rcut , *rcut_max );
      if( grid->number_of_cells() == 0 ) { return; }

      ComputePairOptionalLocks<false> cp_locks {};
      exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
      auto compute_buf = make_compute_pair_buffer<ComputeBuffer>();
      
      auto avg_acc = grid->field_accessor( *avg_field );
      auto nbh_acc = grid->field_accessor( *nbh_field );
      AverageNeighborsFunctor<decltype(avg_acc),decltype(nbh_acc)> compute_op = { (*rcut) * (*rcut) , avg_acc, nbh_acc };
      
      LinearXForm cp_xform { domain->xform() };
      auto optional = make_compute_pair_optional_args( nbh_it, ComputePairNullWeightIterator{} , cp_xform, cp_locks );
      static constexpr onika::FlatTuple<> compute_field_set = {};
      static constexpr std::integral_constant<bool,true> force_use_cells_accessor = {};
      static constexpr DefaultPositionFields posfields = {};
      compute_cell_particle_pairs2( *grid, *rcut, *ghost, optional, compute_buf, compute_op, compute_field_set
                                  , posfields, parallel_execution_context(), force_use_cells_accessor );      
    }

  };

  template<class GridT> using AverageNeighborsScalar = AverageNeighbors<GridT,field::generic_real,field::generic_real>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT(average_neighbors_scalar)
  {  
    OperatorNodeFactory::instance()->register_factory( "average_neighbors_scalar" , make_grid_variant_operator< AverageNeighborsScalar > );
  }

}
