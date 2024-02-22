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

  /************************************************
   *** Chunk neighbors traversal implementation ***
   ************************************************/
  template< class CellT
          , class ComputePairBufferFactoryT, class OptionalArgsT, class FuncT
          , class FieldAccTupleT, class PosFieldsT
          , bool PreferComputeBuffer
          , size_t ... FieldIndex >
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
    onika::IntConst<1> CS,
    bool Symmetric,
    PosFieldsT ,
    onika::BoolConst<PreferComputeBuffer> ,
    std::index_sequence<FieldIndex...> ,
    std::integral_constant<NbhIteratorKind,NbhIteratorKind::CHUNK_NEIGHBORS_RAE> nbh_kind = {}
    )
  {
    static constexpr bool requires_block_synchronous_call = compute_pair_traits::requires_block_synchronous_call_v<FuncT>;

    static_assert( nbh_kind.value == NbhIteratorKind::CHUNK_NEIGHBORS_RAE );
    static_assert( ! requires_block_synchronous_call ); // this implementation doesn't support this

    using exanb::chunknbh_stream_to_next_particle;
    using exanb::chunknbh_stream_info;
    using exanb::decode_cell_index;
  
    [[maybe_unused]] static constexpr bool has_locks = ! std::is_same_v< decltype(optional.locks) , ComputePairOptionalLocks<false> >;
    
    // particle compute context, only for functors not using compute buffer
    static constexpr bool has_particle_start = compute_pair_traits::has_particle_context_start_v<FuncT>;
    static constexpr bool has_particle_stop  = compute_pair_traits::has_particle_context_stop_v<FuncT>;
    static constexpr bool has_particle_ctx   = compute_pair_traits::has_particle_context_v<FuncT>;
    
    static constexpr bool use_compute_buffer = (PreferComputeBuffer || has_particle_start || has_particle_stop) && compute_pair_traits::compute_buffer_compatible_v<FuncT>;
    static_assert( use_compute_buffer || ( ! requires_block_synchronous_call ) , "incompatible functor configuration" );

    using NbhFields = typename OptionalArgsT::nbh_field_tuple_t;
    static constexpr size_t nbh_fields_count = onika::tuple_size_const_v< NbhFields >;
    static_assert( nbh_fields_count==0 || use_compute_buffer , "Neighbor field auto packaging is only supported when using compute bufer by now." );

    //using CLoopBoolT = std::conditional_t< Symetric , bool , onika::BoolConst<true> >;
    using ComputePairBufferT = std::conditional_t< use_compute_buffer || has_particle_ctx || has_particle_start || has_particle_stop, typename ComputePairBufferFactoryT::ComputePairBuffer , onika::BoolConst<false> >;
    
    //static constexpr bool requires_nbh_optional_data = compute_pair_traits::requires_nbh_optional_data_v<FuncT>;
    
    using _Rx = typename PosFieldsT::Rx;
    using _Ry = typename PosFieldsT::Ry;
    using _Rz = typename PosFieldsT::Rz;
    static constexpr onika::soatl::FieldId<_Rx> RX;
    static constexpr onika::soatl::FieldId<_Ry> RY;
    static constexpr onika::soatl::FieldId<_Rz> RZ;

    // cell filtering, allows user to give a selection function enabling or inhibiting cell processing
    if constexpr ( ! std::is_same_v< ComputePairTrivialCellFiltering , std::remove_reference_t<decltype(optional.cell_filter)> > )
    {
      if( ! optional.cell_filter(cell_a,loc_a) ) return;
    }

    // number of particles in cell
    const unsigned int cell_a_particles = cells[cell_a].size();
    const int dims_i = dims.i;
    const int dims_j = dims.j;

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

    const auto stream_info = chunknbh_stream_info( optional.nbh.m_nbh_streams[cell_a] , cell_a_particles );
    const uint16_t* stream_base = stream_info.stream;
    const uint16_t* __restrict__ stream = stream_base;
    const uint32_t* __restrict__ particle_offset = stream_info.offset;
    const uint32_t* __restrict__ sym_nbh_count = stream_info.dual_offset;
    const int32_t poffshift = stream_info.shift;

    assert( particle_offset != nullptr );
    assert( !Symmetric || sym_nbh_count != nullptr );

    unsigned int p_a = ONIKA_CU_THREAD_IDX;

    while( p_a < cell_a_particles )
    {
      // filtering needed because we use unguarded version of ONIKA_CU_BLOCK_SIMD_FOR which doesn't guarantee that p_a<cell_a_particles
      while( p_a<cell_a_particles && !optional.particle_filter(cell_a,p_a) )
      {
        p_a += ONIKA_CU_BLOCK_SIZE ;
      }
      
      if( p_a < cell_a_particles )
      {
        const unsigned int n_neighbors = Symmetric ? sym_nbh_count[p_a] : ( particle_offset[p_a+1] - particle_offset[p_a] );
        stream = stream_base + particle_offset[p_a] + poffshift;
        
        if constexpr ( use_compute_buffer )
        {
          tab.part = p_a;
          tab.count = 0;
        }

        if constexpr ( has_particle_start )
        {
          func(tab,cells,cell_a,p_a,ComputePairParticleContextStart{});
        }

        for(unsigned int p_nbh_index = 0; p_nbh_index<n_neighbors; p_nbh_index++ )
        {
          const uint16_t cell_b_enc = stream[ p_nbh_index * 2 + 0 ];
          const uint16_t p_b        = stream[ p_nbh_index * 2 + 1 ];

          assert( cell_b_enc >= GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE );
          const int rel_i = int( cell_b_enc & 31 ) - 16;
          const int rel_j = int( (cell_b_enc>>5) & 31 ) - 16;
          const int rel_k = int( (cell_b_enc>>10) & 31 ) - 16;
          size_t cell_b = cell_a + ( ( ( rel_k * dims_j ) + rel_j ) * dims_i + rel_i );
          
          const auto * __restrict__ rx_b = cells[cell_b][RX]; 
          const auto * __restrict__ ry_b = cells[cell_b][RY]; 
          const auto * __restrict__ rz_b = cells[cell_b][RZ]; 

          const Vec3d dr = optional.xform.transformCoord( Vec3d{ rx_b[p_b] - rx_a[p_a] , ry_b[p_b] - ry_a[p_a] , rz_b[p_b] - rz_a[p_a] } );
          const double d2 = norm2(dr);
          if( d2>0.0 && d2 <= rcut2 )
          {
#           define NBH_OPT_DATA_ARG optional.nbh_data.get(cell_a, p_a, p_nbh_index, nbh_data_ctx)
            if constexpr ( use_compute_buffer )
            {
              tab.check_buffer_overflow();
              if constexpr ( nbh_fields_count > 0 )
              {
                compute_cell_particle_pairs_pack_nbh_fields( tab , cells , cell_b, p_b, optional.nbh_fields , std::make_index_sequence<nbh_fields_count>{} );
              }
              tab.process_neighbor(tab, dr, d2, cells, cell_b, p_b, NBH_OPT_DATA_ARG );
            }
            if constexpr ( ! use_compute_buffer )
            {
              if constexpr ( has_particle_ctx )
                func( tab, dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, NBH_OPT_DATA_ARG );
              if constexpr ( !has_particle_ctx )
                func(      dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, NBH_OPT_DATA_ARG );                
            }
#           undef NBH_OPT_DATA_ARG
          }
        }

        if constexpr ( use_compute_buffer )
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
      
      p_a += ONIKA_CU_BLOCK_SIZE ;
    } // end of ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED for loop

  }
  /*** end of chunk version implementation ***/    

}

