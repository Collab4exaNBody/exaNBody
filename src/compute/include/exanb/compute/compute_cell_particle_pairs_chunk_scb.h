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
#include <onika/lambda_tools.h>

namespace exanb
{

  /************************************************
   *** Chunk neighbors traversal implementation ***
   ************************************************/
  template< class CellT, unsigned int CS
          , class ComputePairBufferFactoryT, class OptionalArgsT, class FuncT
          , class FieldAccTupleT, class PosFieldsT
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
    onika::UIntConst<CS> nbh_chunk_size ,
    ComputeParticlePairOpts< false, true , true > ,
    PosFieldsT pos_fields ,
    std::index_sequence<FieldIndex...>
    )
  {
    using exanb::chunknbh_stream_info;
    using onika::cuda::min;

    using ComputePairBufferT = typename ComputePairBufferFactoryT::ComputePairBuffer;
    using NbhFields = typename OptionalArgsT::nbh_field_tuple_t;
    static constexpr size_t nbh_fields_count = onika::tuple_size_const_v< NbhFields >;
        
    const auto RX = pos_fields.e0;
    const auto RY = pos_fields.e1;
    const auto RZ = pos_fields.e2;

    // cell filtering, allows user to give a selection function enabling or inhibiting cell processing
    if constexpr ( ! std::is_same_v< ComputePairTrivialCellFiltering , std::remove_reference_t<decltype(optional.cell_filter)> > )
    {
      if( ! optional.cell_filter(cell_a,loc_a) ) return;
    }

    // number of particles in cell
    const unsigned int cell_a_particles = cells[cell_a].size();
    if( cell_a_particles == 0 ) return;
    
    const int dims_i = dims.i;
    const int dims_j = dims.j;

    // spin locks for concurent access
    auto& cell_a_locks = optional.locks[cell_a];
    [[maybe_unused]] auto nbh_data_ctx = optional.nbh_data.make_ctx();

    // create local computation scratch buffer
    ONIKA_CU_BLOCK_SHARED onika::cuda::UnitializedPlaceHolder<ComputePairBufferT> tab_place_holder;

    // pointers to central particle's coordinates
    ONIKA_CU_BLOCK_SHARED const double* __restrict__ rx_a;
    ONIKA_CU_BLOCK_SHARED const double* __restrict__ ry_a;
    ONIKA_CU_BLOCK_SHARED const double* __restrict__ rz_a;

    // chunk neighbors data stream
    ONIKA_CU_BLOCK_SHARED const uint16_t* stream_base;
    ONIKA_CU_BLOCK_SHARED const uint32_t* __restrict__ particle_offset;
    ONIKA_CU_BLOCK_SHARED int32_t poffshift;

    // next place where to write valid neighbor particle
    ONIKA_CU_BLOCK_SHARED int valid_nbh_index;

    ComputePairBufferT & tab = tab_place_holder.get_ref();
    if ( ONIKA_CU_THREAD_IDX == 0 )
    {
      cpbuf_factory.init(tab);
      tab.ta = particle_id_codec::MAX_PARTICLE_TYPE; // not used for single material computations
      tab.tb = particle_id_codec::MAX_PARTICLE_TYPE;
      tab.cell = cell_a;
      
      rx_a = cells[cell_a].field_pointer_or_null(RX);
      ry_a = cells[cell_a].field_pointer_or_null(RY);
      rz_a = cells[cell_a].field_pointer_or_null(RZ);
      
      const auto stream_info = chunknbh_stream_info( optional.nbh.m_nbh_streams[cell_a] , cell_a_particles );
      stream_base = stream_info.stream;
      particle_offset = stream_info.offset;
      poffshift = stream_info.shift;
    }
    ONIKA_CU_BLOCK_SYNC();

    // initialize stream cursor
    const uint16_t* __restrict__ stream = stream_base;

    // pointers to neighbor particle's coordinates
    const double* __restrict__ rx_b = nullptr;
    const double* __restrict__ ry_b = nullptr;
    const double* __restrict__ rz_b = nullptr;
    
    // decode neighbors sream variables
    unsigned int stream_last_p_a = 0;
    int cell_groups = 0;
    int nchunks = 0;

    unsigned int p_nbh_index = 0;
    size_t cell_b = 0;
    int cell_b_particles = 0;

    // compute next particle index to process
    unsigned int p_a = 0; //ONIKA_CU_THREAD_IDX;

    // initialize number of cell groups for first particle, only if cell is not empty
    cell_groups = *(stream++);

    // -- jump to next particle to process --
    while( p_a < cell_a_particles && !optional.particle_filter(cell_a,p_a) ) { p_a ++; }
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
      // --- particle processing start code ---
      if ( ONIKA_CU_THREAD_IDX == 0 )
      {
        tab.part = p_a;
        tab.count = 0;
        valid_nbh_index = 0;
      }
      ONIKA_CU_BLOCK_SYNC();
      // --------------------------------------
      
      while( cell_groups>0 || nchunks>0 )
      {

        if( nchunks == 0 )
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
        
        if( nchunks > 0 )
        {
          const unsigned int stream_chunk_offset = ONIKA_CU_THREAD_IDX;
          const bool is_nbh_chunk_valid = ( stream_chunk_offset < nchunks );
          const unsigned int chunk_b = is_nbh_chunk_valid ? stream[stream_chunk_offset] : 0 ;
          const unsigned int chunks_consumed = ( ONIKA_CU_BLOCK_SIZE >= nchunks ) ? nchunks : ONIKA_CU_BLOCK_SIZE ;
          nchunks -= chunks_consumed;
          stream += chunks_consumed;
          if( is_nbh_chunk_valid )
          {
            const int p_b_start = chunk_b * CS;
            const int p_b_end = min( int( (chunk_b+1) * CS ) , int(cell_b_particles) );
            for( int p_b=p_b_start, tl_nbh_index=0 ; p_b<p_b_end ; p_b++, tl_nbh_index++ )
            {
              const Vec3d dr = optional.xform.transformCoord( Vec3d{ rx_b[p_b] - rx_a[p_a] , ry_b[p_b] - ry_a[p_a] , rz_b[p_b] - rz_a[p_a] } );
              const double d2 = norm2(dr);          
              if( d2>0.0 && d2 <= rcut2 )
              {            
                int tl_nbh_write_idx = ONIKA_CU_ATOMIC_ADD( valid_nbh_index , 1 );
                assert( tl_nbh_write_idx < tab.MaxNeighbors );
                if constexpr ( nbh_fields_count > 0 )
                {
                  compute_cell_particle_pairs_pack_nbh_fields( tab , tl_nbh_write_idx , cells , cell_b, p_b, optional.nbh_fields , std::make_index_sequence<nbh_fields_count>{} );
                }
                tab.process_neighbor(tab, tl_nbh_write_idx , dr, d2, cells, cell_b, p_b, optional.nbh_data.get(cell_a, p_a, p_nbh_index + ( stream_chunk_offset * CS ) + tl_nbh_index , nbh_data_ctx) );
              }
            }
            p_nbh_index += chunks_consumed * CS;            
          }
          
        }

      }

      ONIKA_CU_BLOCK_SYNC();
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        tab.count = valid_nbh_index;
      }
      ONIKA_CU_BLOCK_SYNC();

      // --- particle processing end ---
      static constexpr bool callable_with_locks = onika::lambda_is_callable_with_args_v<FuncT,decltype(tab.count),decltype(tab),decltype(cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part]) ... , decltype(cells) , decltype(optional.locks) , decltype(cell_a_locks[tab.part]) >;
      static constexpr bool trivial_locks = std::is_same_v< decltype(optional.locks) , ComputePairOptionalLocks<false> >;
      static constexpr bool call_with_locks = !trivial_locks && callable_with_locks;    
      if( tab.count > 0 )
      {
        if constexpr ( call_with_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells , optional.locks , cell_a_locks[tab.part] );
        if constexpr (!call_with_locks ) func( tab.count, tab, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][tab.part] ... , cells );
      }
      // --------------------------------
      
      // -- jump to next particle to process --
      p_a ++;
      while( p_a < cell_a_particles && !optional.particle_filter(cell_a,p_a) ) { p_a ++; }
      while( p_a < cell_a_particles && p_a > stream_last_p_a )
      {
        if( particle_offset != nullptr ) { stream = stream_base + particle_offset[p_a] + poffshift; stream_last_p_a = p_a; }
        else { stream += nchunks; while( cell_groups > 0 ) { stream += 2 + stream[1]; -- cell_groups; } nchunks = 0; ++ stream_last_p_a; }
        cell_groups = *(stream++);
      }
      nchunks = 0; p_nbh_index = 0; cell_b = 0;
      // -------------------------------------
      
    } // end of ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED for loop

  }
  /*** end of chunk version implementation ***/    

}

