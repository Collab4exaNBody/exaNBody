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
#include <onika/cuda/cuda_math.h>
#include <onika/lambda_tools.h>

namespace exanb
{

  /************************************************
   *** Chunk neighbors traversal implementation ***
   ************************************************/
  template< class CellT, unsigned int CS, bool Symmetric
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
    onika::UIntConst<CS> nbh_chunk_size,
    onika::BoolConst<Symmetric> ,
    PosFieldsT pos_fields,
    onika::BoolConst<PreferComputeBuffer> ,
    std::index_sequence<FieldIndex...> )
  {
    static constexpr bool requires_block_synchronous_call = compute_pair_traits::requires_block_synchronous_call_v<FuncT>;

    using exanb::chunknbh_stream_info;
    using onika::cuda::min;
      
    // particle compute context, only for functors not using compute buffer
    static constexpr bool has_particle_start = compute_pair_traits::has_particle_context_start_v<FuncT>;
    static constexpr bool has_particle_stop  = compute_pair_traits::has_particle_context_stop_v<FuncT>;
    static constexpr bool has_particle_ctx   = compute_pair_traits::has_particle_context_v<FuncT>;
    
    static constexpr bool use_compute_buffer = PreferComputeBuffer && compute_pair_traits::compute_buffer_compatible_v<FuncT>;
    static_assert( use_compute_buffer || ( ! requires_block_synchronous_call ) , "incompatible functor configuration" );

    using NbhFields = typename OptionalArgsT::nbh_field_tuple_t;
    static constexpr size_t nbh_fields_count = onika::tuple_size_const_v< NbhFields >;
    static_assert( nbh_fields_count==0 || use_compute_buffer , "Neighbor field auto packaging is only supported when using compute buffer by now." );

    using ComputePairBufferT = std::conditional_t< use_compute_buffer 
                                                 , typename ComputePairBufferFactoryT::ComputePairBuffer
                                                 , std::conditional_t< has_particle_ctx || has_particle_start || has_particle_stop
                                                                     , typename ComputePairBufferFactoryT::ComputePairBuffer::ContextWithoutBuffer
                                                                     , onika::BoolConst<false>
                                                                     >
                                                 >;

    const auto RX = pos_fields.e0;
    const auto RY = pos_fields.e1;
    const auto RZ = pos_fields.e2;

    // cell filtering, allows user to give a selection function enabling or inhibiting cell processing
    if constexpr ( ! std::is_same_v< ComputePairTrivialCellFiltering , std::remove_reference_t<decltype(optional.cell_filter)> > )
    {
      if( ! optional.cell_filter(cell_a,loc_a) ) return;
    }

    // number of particles in cell
    const int cell_a_particles = cells[cell_a].size();
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

    // without compute buffer, but with delayed computation buffer to resynchronize GPU thread computation phases
    static constexpr unsigned int DELAYED_COMPUTE_BUFFER_SIZE = ( !use_compute_buffer && gpu_device_execution() ) ? XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE : 1; // must be one and not 0 to avoid GCC compiler warnings about %0, even if it is in 'if constexpr' block
    using DelayedComputeBufferT = std::conditional_t< (DELAYED_COMPUTE_BUFFER_SIZE>1) , DelayedComputeBuffer<DELAYED_COMPUTE_BUFFER_SIZE*XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE> , onika::BoolConst<false> >;
    [[maybe_unused]] ONIKA_CU_BLOCK_SHARED DelayedComputeBufferT dcb;
#   define DNCB(i) dcb.nbh[i*XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE+ONIKA_CU_THREAD_IDX]
    [[maybe_unused]] unsigned int dncb_start = 0;
    [[maybe_unused]] unsigned int dncb_free = DELAYED_COMPUTE_BUFFER_SIZE;

    // pointers to central particle's coordinates
    const double* __restrict__ rx_a = cells[cell_a].field_pointer_or_null(RX);
    const double* __restrict__ ry_a = cells[cell_a].field_pointer_or_null(RY);
    const double* __restrict__ rz_a = cells[cell_a].field_pointer_or_null(RZ);

    // chunk neighbors data stream
    const auto stream_info = chunknbh_stream_info( optional.nbh.m_nbh_streams[cell_a] , cell_a_particles );
    const uint16_t* stream_base = stream_info.stream;
    const uint16_t* __restrict__ stream = stream_base;
    const uint32_t* __restrict__ particle_offset = stream_info.offset;
    const int32_t poffshift = stream_info.shift;

    // pointers to neighbor particle's coordinates
    const double* __restrict__ rx_b = nullptr;
    const double* __restrict__ ry_b = nullptr;
    const double* __restrict__ rz_b = nullptr;
    
    // decode neighbor stream variables
    int stream_last_p_a = 0;
    int cell_groups = 0;
    int nchunks = 0;
    int chunk_particles = 0;

    unsigned int p_nbh_index = 0;
    size_t cell_b = 0;
    int p_b = 0;
    int cell_b_particles = 0;

    // compute next particle index to process
    int p_a = ONIKA_CU_THREAD_IDX;

    // initialize number of cell groups for first particle, only if cell is not empty
    if( cell_a_particles > 0 ) cell_groups = *(stream++);

    // -- jump to next particle to process --
    while( p_a < cell_a_particles && !optional.particle_filter(cell_a,p_a) ) { p_a += ONIKA_CU_BLOCK_SIZE; }
    while( p_a < cell_a_particles && p_a > stream_last_p_a )
    {
      if( particle_offset != nullptr ) { stream = stream_base + particle_offset[p_a] + poffshift; stream_last_p_a = p_a; }
      else { stream += nchunks; while( cell_groups > 0 ) { stream += 2 + stream[1]; -- cell_groups; } nchunks = 0; ++ stream_last_p_a; }
      cell_groups = *(stream++);
    }
    nchunks = 0; p_nbh_index = 0; cell_b = 0;
    // -------------------------------------

    while( p_a < cell_a_particles )
    {
      //printf("cell %05d , p %05d , r=%g,%g,%g\n",int(cell_a),int(p_a),rx_a[p_a],ry_a[p_a],rz_a[p_a]);

      // --- particle processing start code ---
      if constexpr ( use_compute_buffer ) { tab.part = p_a; tab.count = 0; }
      if constexpr ( has_particle_start ) { func(tab,cells,cell_a,p_a,ComputePairParticleContextStart{}); }
      // --------------------------------------
      
      while( cell_groups>0 || nchunks>0 || chunk_particles>0 )
      {
        //printf("%d / %d / %d\n",int(cell_groups),int(nchunks),int(chunk_particles));
        if( nchunks == 0 && chunk_particles == 0 )
        {
          if( cell_groups > 0 )
          {
            // restart from next cell group
            const uint16_t cell_b_enc = *(stream++);
            assert( cell_b_enc >= GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE );
            nchunks = *(stream++); assert( nchunks > 0 );
            const int rel_i = int( cell_b_enc & 31 ) - 16;
            const int rel_j = int( (cell_b_enc>>5) & 31 ) - 16;
            const int rel_k = int( (cell_b_enc>>10) & 31 ) - 16;
            cell_b = cell_a + ( ( ( rel_k * dims_j ) + rel_j ) * dims_i + rel_i );
            cell_b_particles = cells[cell_b].size();            
            rx_b = cells[cell_b].field_pointer_or_null(RX);
            ry_b = cells[cell_b].field_pointer_or_null(RY);
            rz_b = cells[cell_b].field_pointer_or_null(RZ);
            -- cell_groups;
          }
        }
        
        if( nchunks > 0 || chunk_particles > 0 )
        {
        
          if( chunk_particles == 0 )
          {
            p_b = static_cast<unsigned int>( *(stream++) ) * CS;
            -- nchunks;
            chunk_particles = min( int(CS) , cell_b_particles - p_b );
          }    
          
          if constexpr ( Symmetric ) if( cell_b > cell_a || ( cell_b == cell_a && p_b > p_a ) ) break;
          
          if( cell_a!=cell_b || p_a!=p_b )
          {
            const Vec3d dr = optional.xform.transformCoord( Vec3d{ rx_b[p_b] - rx_a[p_a] , ry_b[p_b] - ry_a[p_a] , rz_b[p_b] - rz_a[p_a] } );
            const double d2 = norm2(dr);
            if( d2>0.0 && d2 <= rcut2 )
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
                if constexpr ( DELAYED_COMPUTE_BUFFER_SIZE > 1 )
                {
                  assert( dncb_free > 0 );
                  const auto dncb_end = ( dncb_start + DELAYED_COMPUTE_BUFFER_SIZE - dncb_free ) % DELAYED_COMPUTE_BUFFER_SIZE ;
                  DNCB(dncb_end) = { dr.x,dr.y,dr.z, d2, static_cast<uint32_t>(cell_b), static_cast<uint16_t>(p_b), static_cast<uint16_t>(p_nbh_index) };
                  -- dncb_free;
                }
                else
                {
                  if constexpr ( has_particle_ctx )
                    func( tab, dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get(cell_a, p_a, p_nbh_index, nbh_data_ctx) );
                  if constexpr ( !has_particle_ctx )
                    func(      dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get(cell_a, p_a, p_nbh_index, nbh_data_ctx) );                
                }
              }
            }
            ++ p_nbh_index;
          }
          ++ p_b;
          -- chunk_particles;      
        }

        if constexpr ( !use_compute_buffer && DELAYED_COMPUTE_BUFFER_SIZE>1 )
        {
          if( ONIKA_CU_WARP_ACTIVE_THREADS_ANY( dncb_free == 0 ) )
          {
            if( dncb_free < DELAYED_COMPUTE_BUFFER_SIZE )
            {
              const Vec3d dr = { DNCB(dncb_start).drx , DNCB(dncb_start).dry , DNCB(dncb_start).drz };
              if constexpr ( has_particle_ctx )
                func( tab, dr, DNCB(dncb_start).d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells
                    , DNCB(dncb_start).cell_b, DNCB(dncb_start).p_b, optional.nbh_data.get(cell_a, p_a, DNCB(dncb_start).p_nbh_index, nbh_data_ctx) );
              if constexpr ( !has_particle_ctx )
                func(      dr, DNCB(dncb_start).d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells 
                    , DNCB(dncb_start).cell_b, DNCB(dncb_start).p_b, optional.nbh_data.get(cell_a, p_a, DNCB(dncb_start).p_nbh_index, nbh_data_ctx) );                
              dncb_start = ( dncb_start + 1 ) % DELAYED_COMPUTE_BUFFER_SIZE;
              ++ dncb_free;
            }
          }
        }
      }

      // --- particle processing end ---
      if constexpr ( use_compute_buffer )
      {
        static constexpr bool callable_with_locks = onika::lambda_is_callable_with_args_v<FuncT,decltype(tab.count),decltype(tab),decltype(cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part]) ... , decltype(cells) , decltype(optional.locks) , decltype(cell_a_locks[tab.part]) >;
        static constexpr bool trivial_locks = std::is_same_v< decltype(optional.locks) , ComputePairOptionalLocks<false> >;
        static constexpr bool call_with_locks = !trivial_locks && callable_with_locks;    
        if( tab.count > 0 || requires_block_synchronous_call ) 
        {
          if constexpr ( call_with_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells , optional.locks , cell_a_locks[tab.part] );
          if constexpr (!call_with_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells );
        }
      }

      if constexpr ( !use_compute_buffer && DELAYED_COMPUTE_BUFFER_SIZE>1 )
      {
        while( dncb_free < DELAYED_COMPUTE_BUFFER_SIZE )
        {
          const Vec3d dr = { DNCB(dncb_start).drx , DNCB(dncb_start).dry , DNCB(dncb_start).drz };
          if constexpr ( has_particle_ctx )
            func( tab, dr, DNCB(dncb_start).d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells
                , DNCB(dncb_start).cell_b, DNCB(dncb_start).p_b, optional.nbh_data.get(cell_a, p_a, DNCB(dncb_start).p_nbh_index, nbh_data_ctx) );
          if constexpr ( !has_particle_ctx )
            func(      dr, DNCB(dncb_start).d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells 
                , DNCB(dncb_start).cell_b, DNCB(dncb_start).p_b, optional.nbh_data.get(cell_a, p_a, DNCB(dncb_start).p_nbh_index, nbh_data_ctx) );                
          dncb_start = ( dncb_start + 1 ) % DELAYED_COMPUTE_BUFFER_SIZE;
          ++ dncb_free;
        }
        assert( dncb_free == DELAYED_COMPUTE_BUFFER_SIZE );
        dncb_start = 0;
      }
      
      if constexpr ( has_particle_stop ) { func(tab ,cells,cell_a,p_a,ComputePairParticleContextStop{}); }
      // --------------------------------
      
      // -- jump to next particle to process --
      p_a += ONIKA_CU_BLOCK_SIZE ;
      while( p_a < cell_a_particles && !optional.particle_filter(cell_a,p_a) ) { p_a += ONIKA_CU_BLOCK_SIZE; }
      while( p_a < cell_a_particles && p_a > stream_last_p_a )
      {
        if( particle_offset != nullptr ) { stream = stream_base + particle_offset[p_a] + poffshift; stream_last_p_a = p_a; }
        else { stream += nchunks; while( cell_groups > 0 ) { stream += 2 + stream[1]; -- cell_groups; } nchunks = 0; ++ stream_last_p_a; }
        cell_groups = *(stream++);
      }
      nchunks = 0; chunk_particles=0; p_nbh_index = 0; cell_b = 0;
      // -------------------------------------
      
    } // end of ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED for loop

  }
  /*** end of chunk version implementation ***/    
# undef DNCB

}

