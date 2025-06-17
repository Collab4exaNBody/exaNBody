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
#include <exanb/core/particle_type_properties.h>

// this allows for parallel compilation of templated operator for each available field set
namespace exanb
{

  // AverageNeighbors Context
  template<class ValueT>
  struct AverageNeighborsExtStorage
  {
    ValueT m_sum = {};
    double m_weight_sum = 0.0;
    
    ONIKA_HOST_DEVICE_FUNC
    inline void reset()
    {
      m_sum = ValueT{};
      m_weight_sum = 0.0;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline auto avg() const
    {
      return ( m_weight_sum > 0.0 ) ? ( m_sum / m_weight_sum ) : ( m_sum / 1.0 );
    }
  };

  // interaction potential compute functor
  template<class AvgFieldT, class NbhFieldT>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) AverageNeighborsFunctor
  {
    // potential function parameters
    const double m_rcut_sq = {};
    const double a0 = 1.0;
    const double a1 = 0.0;
    const double a2 = 0.0;
    const double a3 = 0.0;
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
        auto w = a0 + a2*d2;
        if( a1!=0.0 || a3!=0.0 ) { const auto d = sqrt(d2); w += a1*d + a3*d2*d; }
        ctx.ext.m_sum += w * cells[cell_b][m_nbh_field][p_b];
        ctx.ext.m_weight_sum += w;
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

  template<class GridT>
  class AverageNeighborsScalar : public OperatorNode
  {
    using DoubleVector = onika::memory::CudaMMVector<double>;
    
    // ========= I/O slots =======================
    ADD_SLOT( double                    , rcut            , INPUT        , 0.0 , DocString{"Cutoff distance for average operation."} );
    ADD_SLOT( DoubleVector              , weight_function , INPUT        , DoubleVector{1.0} , DocString{"List of [a0,...,an] coefficients for the polynomial distance weighting function : a0*x^0 + a1*x^1 + ... +an*x^n"} );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors , INPUT        , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
    ADD_SLOT( Domain                    , domain          , INPUT        , REQUIRED , DocString{"Simulation domain"});
    ADD_SLOT( ParticleTypeProperties    , particle_type_properties , INPUT , ParticleTypeProperties{} );

    ADD_SLOT( std::string               , avg_field       , INPUT        , REQUIRED , DocString{"Name of the resulting averaged field."});
    ADD_SLOT( std::string               , nbh_field       , INPUT        , REQUIRED , DocString{"Name of the neighbors field to be averaged."});

    ADD_SLOT( double                    , rcut_max        , INPUT_OUTPUT , 0.0 , DocString{"Updated max rcut"});
    ADD_SLOT( GridT                     , grid            , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );

    template<class FieldT>
    inline void test_and_execute_input_field( const FieldT& input_field ) 
    {
      if( (*nbh_field) == input_field.short_name() )
      {
        using ComputeBuffer = ComputePairBuffer2<false,false,AverageNeighborsExtStorage<typename FieldT::value_type> >;

        ComputePairOptionalLocks<false> cp_locks {};
        exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
        auto compute_buf = make_compute_pair_buffer<ComputeBuffer>();
        
        auto avg_acc = grid->field_accessor( field::mk_generic_real( *avg_field ) );
        auto nbh_acc = grid->field_const_accessor( input_field );
        double poly_coefs[4] = { 1.0 , 0.0 , 0.0 , 0.0 };
        for(size_t i=0;i<weight_function->size() && i<4;i++) poly_coefs[i] = weight_function->at(i);
        AverageNeighborsFunctor<decltype(avg_acc),decltype(nbh_acc)> compute_op = { (*rcut) * (*rcut) , poly_coefs[0] , poly_coefs[1] , poly_coefs[2] , poly_coefs[3] , avg_acc, nbh_acc };

        LinearXForm cp_xform { domain->xform() };
        auto optional = make_compute_pair_optional_args( nbh_it, ComputePairNullWeightIterator{} , cp_xform, cp_locks );
        static constexpr onika::FlatTuple<> compute_field_set = {};
        static constexpr std::integral_constant<bool,true> force_use_cells_accessor = {};
        static constexpr DefaultPositionFields posfields = {};
        compute_cell_particle_pairs2( *grid, *rcut, false, optional, compute_buf, compute_op, compute_field_set
                                    , posfields, parallel_execution_context(), force_use_cells_accessor );
/*
        const auto cells = grid->cells_accessor();
        for(size_t c=0;c<grid->number_of_cells();c++)
        {
          size_t n = cells[c].size();
          lout << "Cell #"<<c<<" :";
          for(size_t p=0;p<n;p++) { lout << " " << cells[c][avg_acc][p]; }
          lout << std::endl;
        }
*/
      }
    }

    template<class FieldT>
    inline void test_and_execute_input_field( const std::span<FieldT>& input_fields ) 
    {
      for(const auto& f : input_fields) test_and_execute_input_field( f );
    }

    template<class... GridFields>
    inline void test_all_input_fields( const GridFields& ... grid_fields ) 
    {
      ( ... , ( test_and_execute_input_field(grid_fields) ) );
    }

    template<class... fid>
    inline void execute_on_field_set( FieldSet<fid...> ) 
    {
      using has_field_type_t = typename GridT:: template HasField < field::_type >;
      static constexpr bool has_field_type = has_field_type_t::value;
      
      std::vector< TypePropertyScalarCombiner > type_scalars;
      if constexpr ( has_field_type )
      {
        if( particle_type_properties.has_value() )
        {
          for(const auto & it : particle_type_properties->m_scalars)
          {
            type_scalars.push_back( { make_type_property_functor( it.first , it.second.data() ) } );
          }
        }
      }
      std::span<TypePropertyScalarCombiner> particle_type_fields = type_scalars;

      test_all_input_fields( particle_type_fields , onika::soatl::FieldId<fid>{} ... );
    }

  public:
    // Operator execution
    inline void execute () override final
    {
      assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );
      *rcut_max = std::max( *rcut , *rcut_max );
      if( grid->number_of_cells() == 0 ) return;
      if( weight_function->size() > 4 )
      {
        fatal_error()<<"weighting function polynomial has a maximum degree of 3 (maximum 4 coefficients)"<<std::endl;
      }
      using GridFieldSet = typename GridT::field_set_t ;
      execute_on_field_set( GridFieldSet{} );
    }

    inline std::string documentation() const override final
    {
      return R"EOF(

Averaging a per-particle scalar field over neighboring particles in a sphere with a user-specified cutoff radius. Both the field to be average (nbh_field) and the resulting one (avg_field) are user-specified strings.

Usage example:

average_neighbors_scalar:
  nbh_field: mass
  avg_field: avg_mass
  rcut: 8.0 ang
  weight_function: [ 1.0 , 0.0 , -0.01] # => 1 + 0.0 r - 0.01 r^2, r being the distance to the central particle

average_neighbors_scalar:
  nbh_field: charge
  avg_field: avg_charge
  rcut: 12.0 ang
  weight_function: [ 1.0 , 0.0 , -0.01] # => 1 + 0.0 r - 0.01 r^2, r being the distance to the central particle

)EOF";
    }    
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(average_neighbors_scalar)
  {  
    OperatorNodeFactory::instance()->register_factory( "average_neighbors_scalar" , make_grid_variant_operator< AverageNeighborsScalar > );
  }

}
