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

#include <exanb/compute/compute_cell_particle_pairs_common.h>

namespace exanb
{

  /*************************************************
   *** Simple neighbors traversal implementation ***
   *************************************************/
  template< class CellT, class CST, bool Symetric
          , class ComputePairBufferFactoryT, class OptionalArgsT, class FuncT
          , class FieldAccTupleT, class PosFieldsT
          , bool PreferComputeBuffer, size_t ... FieldIndex >
  ONIKA_HOST_DEVICE_FUNC
  static inline void compute_cell_particle_pairs_cell(
    CellT cells,
    IJK dims,
    IJK loc_a,
    size_t cell_a,
    double rcut2,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const OptionalArgsT& optional, // locks are needed if symmetric computation is enabled
    const FuncT& func,
    const FieldAccTupleT& cp_fields ,
    CST ,
    onika::BoolConst<Symetric> ,
    PosFieldsT ,
    onika::BoolConst<PreferComputeBuffer> ,
    std::index_sequence<FieldIndex...> ,
    std::integral_constant<NbhIteratorKind,NbhIteratorKind::SIMPLE_CELL_INDEX> nbh_kind
    )
  {
    static_assert( nbh_kind.value == NbhIteratorKind::SIMPLE_CELL_INDEX );

    using exanb::chunknbh_stream_to_next_particle;
    using exanb::chunknbh_stream_info;
    using exanb::decode_cell_index;
  
    [[maybe_unused]] static constexpr bool has_locks = ! std::is_same_v< decltype(optional.locks) , ComputePairOptionalLocks<false> >;
    
    // particle compute context, only for functors not using compute buffer
    static constexpr bool has_particle_start = compute_pair_traits::has_particle_context_start_v<FuncT>;
    static constexpr bool has_particle_stop  = compute_pair_traits::has_particle_context_stop_v<FuncT>;
    static constexpr bool has_particle_ctx   = compute_pair_traits::has_particle_context_v<FuncT>;
        
    static constexpr bool use_compute_buffer = (PreferComputeBuffer || has_particle_start || has_particle_stop) && compute_pair_traits::compute_buffer_compatible_v<FuncT>;
    static constexpr bool requires_block_synchronous_call = compute_pair_traits::requires_block_synchronous_call_v<FuncT> ;
    static_assert( use_compute_buffer || ( ! requires_block_synchronous_call ) , "incompatible functor configuration" );

    using NbhFields = typename OptionalArgsT::nbh_field_tuple_t;
    static constexpr size_t nbh_fields_count = onika::tuple_size_const_v< NbhFields >;
    static_assert( nbh_fields_count==0 || use_compute_buffer , "Neighbor field auto packaging is only supported when using compute bufer by now." );

    using ComputePairBufferT = std::conditional_t< use_compute_buffer || has_particle_ctx || has_particle_start || has_particle_stop, typename ComputePairBufferFactoryT::ComputePairBuffer , onika::BoolConst<false> >;
    
    using _Rx = typename PosFieldsT::Rx;
    using _Ry = typename PosFieldsT::Ry;
    using _Rz = typename PosFieldsT::Rz;
    static constexpr onika::soatl::FieldId<_Rx> RX;
    static constexpr onika::soatl::FieldId<_Ry> RY;
    static constexpr onika::soatl::FieldId<_Rz> RZ;

    // assert( cells != nullptr );

    // cell filtering, allows user to give a selection function enabling or inhibiting cell processing
    if constexpr ( ! std::is_same_v< ComputePairTrivialCellFiltering , std::remove_reference_t<decltype(optional.cell_filter)> > )
    {
      if( ! optional.cell_filter(cell_a,loc_a) ) return;
    }

    // number of particles in cell
    const unsigned int cell_a_particles = cells[cell_a].size();

    // spin locks for concurent access
    auto& cell_a_locks = optional.locks[cell_a];
    [[maybe_unused]] auto nbh_data_ctx = optional.nbh_data.make_ctx();

    // create local computation scratch buffer
    [[maybe_unused]] ComputePairBufferT tab;
    if constexpr ( use_compute_buffer )
    {
      cpbuf_factory.init(tab);
      tab.ta = particle_id_codec::MAX_PARTICLE_TYPE; // not used for single material computations
      tab.tb = particle_id_codec::MAX_PARTICLE_TYPE;
      tab.cell = cell_a;
    }

    // central particle's coordinates
    const double* __restrict__ rx_a = cells[cell_a][RX]; ONIKA_ASSUME_ALIGNED(rx_a);
    const double* __restrict__ ry_a = cells[cell_a][RY]; ONIKA_ASSUME_ALIGNED(ry_a);
    const double* __restrict__ rz_a = cells[cell_a][RZ]; ONIKA_ASSUME_ALIGNED(rz_a);

    const auto & cell_nbh = onika::cuda::vector_data( optional.nbh.m_cell_neighbors )[cell_a];
    const auto* __restrict__ nbh_offset = onika::cuda::vector_data( cell_nbh.nbh_start );
    const auto* __restrict__ neighbors = onika::cuda::vector_data( cell_nbh.neighbors );
    
    ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED(unsigned int , p_a , 0 , cell_a_particles )
    {
      if( p_a < cell_a_particles && optional.particle_filter(cell_a,p_a) )
      {
        if constexpr ( use_compute_buffer )
        {
          tab.part = p_a;
          tab.count = 0;
        }
        if constexpr ( has_particle_start )
        {
          func(tab,cells,cell_a,p_a,ComputePairParticleContextStart{});
        }
        const int start = ( p_a > 0 ) ? nbh_offset[p_a-1] : 0 ;
        const int end = nbh_offset[p_a];
        const int n_nbh = end - start;
        for( int p_nbh_index=0 ; p_nbh_index<n_nbh ; p_nbh_index++ )
        {
          const auto [ cell_b , p_b ] = neighbors[ start + p_nbh_index ];
          Vec3d dr { cells[cell_b][RX][p_b] - rx_a[p_a] , cells[cell_b][RY][p_b] - ry_a[p_a] , cells[cell_b][RZ][p_b] - rz_a[p_a] };
          dr = optional.xform.transformCoord( dr );
          const double d2 = norm2(dr);
          if( d2 <= rcut2 )
          {
            if constexpr ( use_compute_buffer )
            {
              tab.check_buffer_overflow();
              if constexpr ( nbh_fields_count > 0 )
              {
                compute_cell_particle_pairs_pack_nbh_fields( tab , cells , cell_b, p_b, optional.nbh_fields , std::make_index_sequence<nbh_fields_count>{} );
              }
              tab.process_neighbor(tab, dr, d2, cells, cell_b, p_b, optional.nbh_data.get(cell_a, p_a, p_nbh_index, nbh_data_ctx) );
            }
            if constexpr ( ! use_compute_buffer )
            {
              if constexpr (  has_particle_ctx )
                func( tab, dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get( cell_a , p_a, p_nbh_index , nbh_data_ctx ) );
              if constexpr ( !has_particle_ctx )
                func(      dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get( cell_a , p_a, p_nbh_index , nbh_data_ctx ) );                
            }
          }
        }
        if constexpr ( use_compute_buffer && ! requires_block_synchronous_call )
        {
          if( tab.count > 0 )
          {
            if constexpr ( has_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells , optional.locks , cell_a_locks[tab.part] );
            if constexpr (!has_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells );
          }
        }
        if constexpr ( has_particle_stop )
        {
          func(tab ,cells,cell_a,p_a,ComputePairParticleContextStop{});
        }
                    
      } // CU_BLOCK_SIMD_FOR filtering if
      else 
      {
        if constexpr ( use_compute_buffer && requires_block_synchronous_call )
        {
          // call function with no neighbor particles for SM processor BLOCK parallelism
          tab.count = 0;
          tab.part = 0;
        }
      }

      if constexpr ( use_compute_buffer && requires_block_synchronous_call )
      {
        if constexpr ( has_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells , optional.locks , cell_a_locks[tab.part] );
        if constexpr (!has_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells );
      }
      
    } // end of ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED for loop
            
  }
  /*** end of simple neighbors version implementation ***/

}

