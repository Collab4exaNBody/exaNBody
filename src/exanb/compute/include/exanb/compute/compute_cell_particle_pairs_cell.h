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

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/simple_particle_neighbors.h>

#include <exanb/compute/compute_pair_buffer.h>
#include <exanb/compute/compute_pair_optional_args.h>
#include <exanb/compute/compute_pair_traits.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/field_sets.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/log.h>

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_pointer_tuple.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/integral_constant.h>
//#include <onika/declare_if.h>

#include <cstddef>

namespace exanb
{

  template<class _Rx, class _Ry, class _Rz>
  struct PosititionFields
  {
    using Rx = _Rx;
    using Ry = _Ry;
    using Rz = _Rz;
  };
  using DefaultPositionFields = PosititionFields< field::_rx , field::_ry , field::_rz >;
  
  template<class _Rx, class _Ry, class _Rz>
  ONIKA_HOST_DEVICE_FUNC static inline constexpr PosititionFields<_Rx,_Ry,_Rz> make_position_fields( _Rx , _Ry , _Rz ) { return {}; }
  // compute_cell_particle_pairs( ... , make_position_fields( field::rx , field::ry , field::rz ) )


  template< class ComputeBufferT, class CellsT, class FieldAccessorTupleT, size_t ... FieldIndex >
  ONIKA_HOST_DEVICE_FUNC
  static inline void compute_cell_particle_pairs_pack_nbh_fields( ComputeBufferT& tab, CellsT cells, size_t cell_b, size_t p_b, const FieldAccessorTupleT& nbh_fields , std::index_sequence<FieldIndex...> )
  {
    using NbhTuple = typename ComputeBufferT::NbhFieldTuple;
    tab.nbh_pt[tab.count] = NbhTuple { cells[cell_b][nbh_fields.get(onika::tuple_index_t<FieldIndex>{})][p_b] ... };
  }

  template<class NbhIteratorT> struct NbhIteratorTraits
  {
    static inline constexpr bool is_chunk = false;
    static inline constexpr bool is_simple = false;
  };
  template<bool SYM> struct NbhIteratorTraits< GridChunkNeighborsLightWeightIt<SYM> >
  {
    static inline constexpr bool is_chunk = true;
    static inline constexpr bool is_simple = false;
  };
  template<> struct NbhIteratorTraits< GridSimpleParticleNeighbors >
  {
    static inline constexpr bool is_chunk = false;
    static inline constexpr bool is_simple = true;
  };

  /*
   * FIXME: compact pair weighting structure will need refactoring for cuda compatibility
   */
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
    CST CS /*nbh_chunk_size*/,
    std::integral_constant<bool,Symetric> ,
    PosFieldsT ,
    onika::BoolConst<PreferComputeBuffer> ,
    std::index_sequence<FieldIndex...>
    )
  {
    // static_assert( FieldAccTupleT::size() == sizeof...(FieldIndex) );

    using exanb::chunknbh_stream_to_next_particle;
    using exanb::chunknbh_stream_info;
    using exanb::decode_cell_index;
  
    static constexpr bool has_locks = ! std::is_same_v< decltype(optional.locks) , ComputePairOptionalLocks<false> >;
    //static constexpr bool has_nbh_data = optional.nbh_data.c_has_nbh_data ;
    //static constexpr auto default_nbh_data = optional.nbh_data.c_default_value;
    
    // particle compute context, only for functors not using compute buffer
    static constexpr bool has_particle_start = ComputePairParticleContextTraits<FuncT>::HasParticleContextStart;
    static constexpr bool has_particle_stop = ComputePairParticleContextTraits<FuncT>::HasParticleContextStop;

    static_assert( (!has_particle_start && !has_particle_stop) || ComputePairTraits<FuncT>::ComputeBufferCompatible );
    
    static constexpr bool use_compute_buffer = (PreferComputeBuffer || has_particle_start || has_particle_stop) && ComputePairTraits<FuncT>::ComputeBufferCompatible;
    static constexpr bool requires_block_synchronous_call = ComputePairTraits<FuncT>::RequiresBlockSynchronousCall ;
    static_assert( use_compute_buffer || ( ! requires_block_synchronous_call ) , "incompatible functor configuration" );

    using NbhFields = typename OptionalArgsT::nbh_field_tuple_t;
    static constexpr size_t nbh_fields_count = onika::tuple_size_const_v< NbhFields >;
    static_assert( nbh_fields_count==0 || use_compute_buffer , "Neighbor field auto packaging is only supported when using compute bufer by now." );

    using CLoopBoolT = std::conditional_t< Symetric , bool , onika::BoolConst<true> >;
    using ComputePairBufferT = std::conditional_t< use_compute_buffer , typename ComputePairBufferFactoryT::ComputePairBuffer , onika::BoolConst<false> >;
    
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

    static_assert( NbhIteratorTraits<decltype(optional.nbh)>::is_chunk || NbhIteratorTraits<decltype(optional.nbh)>::is_simple );

    // number of particles in cell
    const unsigned int cell_a_particles = cells[cell_a].size();

    // spin locks for concurent access
    auto& cell_a_locks = optional.locks[cell_a];
    [[maybe_unused]] auto nbh_data_ctx = optional.nbh_data.make_ctx();

    // create local computation scratch buffer
    ComputePairBufferT tab;
    if constexpr ( use_compute_buffer )
    {
      cpbuf_factory.init(tab);
      tab.ta = particle_id_codec::MAX_PARTICLE_TYPE; // not used for single material computations
      tab.tb = particle_id_codec::MAX_PARTICLE_TYPE;
      tab.cell = cell_a;
    }
    if constexpr ( ! use_compute_buffer )
    {
      if constexpr ( tab ) {}
      if constexpr ( has_locks ) {}
    }

    // central particle's coordinates
    const double* __restrict__ rx_a = cells[cell_a][RX]; ONIKA_ASSUME_ALIGNED(rx_a);
    const double* __restrict__ ry_a = cells[cell_a][RY]; ONIKA_ASSUME_ALIGNED(ry_a);
    const double* __restrict__ rz_a = cells[cell_a][RZ]; ONIKA_ASSUME_ALIGNED(rz_a);

    /************************************************
     *** Chunk neighbors traversal implementation ***
     ************************************************/
    if constexpr ( NbhIteratorTraits<decltype(optional.nbh)>::is_chunk )
    {
      const auto stream_info = chunknbh_stream_info( optional.nbh.m_nbh_streams[cell_a] , cell_a_particles );
      const uint16_t* stream_base = stream_info.stream;
      const uint16_t* __restrict__ stream = stream_base;
      const uint32_t* __restrict__ particle_offset = stream_info.offset;
      const int32_t poffshift = stream_info.shift;
      
      //for(unsigned int p_a=0;p_a<cell_a_particles;p_a++)
      //ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , p_a , 0 , cell_a_particles )
      ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED(unsigned int , p_a , 0 , cell_a_particles )
      {
        // filtering needed because we use unguarded version of ONIKA_CU_BLOCK_SIMD_FOR which doesn't guarantee that p_a<cell_a_particles
        if( p_a < cell_a_particles && optional.particle_filter(cell_a,p_a) )
        {
          size_t p_nbh_index = 0;

          if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;
          
          const double* __restrict__ rx_b = nullptr; ONIKA_ASSUME_ALIGNED(rx_b);
          const double* __restrict__ ry_b = nullptr; ONIKA_ASSUME_ALIGNED(ry_b);
          const double* __restrict__ rz_b = nullptr; ONIKA_ASSUME_ALIGNED(rz_b);

          if constexpr ( use_compute_buffer )
          {
            tab.part = p_a;
            tab.count = 0;
          }

          if constexpr ( has_particle_start )
          {
            func(tab,cells,cell_a,p_a,ComputePairParticleContextStart{});
          }

          const unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list
          unsigned int chunk = 0;
          unsigned int nchunks = 0;
          unsigned int cg = 0; // cell group index.
          
          CLoopBoolT symcont={};
          if constexpr (Symetric) symcont = true;
          
          for(cg=0; cg<cell_groups && symcont ;cg++)
          { 
            const uint16_t cell_b_enc = *(stream++);
            assert( cell_b_enc >= GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE );
            const IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
            assert( loc_b.i>=0 && loc_b.j>=0 && loc_b.j>=0 && loc_b.i<dims.i && loc_b.j<dims.j && loc_b.j<dims.j );
            const size_t cell_b = grid_ijk_to_index( dims , loc_b );
            const unsigned int nbh_cell_particles = cells[cell_b].size();
            rx_b = cells[cell_b][RX]; ONIKA_ASSUME_ALIGNED(rx_b);
            ry_b = cells[cell_b][RY]; ONIKA_ASSUME_ALIGNED(ry_b);
            rz_b = cells[cell_b][RZ]; ONIKA_ASSUME_ALIGNED(rz_b);
            nchunks = *(stream++);
            for(chunk=0;chunk<nchunks && symcont;chunk++)
            {
              const unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
              for(unsigned int i=0;i<CS && symcont;i++)
              {
                const unsigned int p_b = chunk_start + i;
                if( Symetric && ( cell_b>cell_a || ( cell_b==cell_a && p_b>=p_a ) ) )
                {
                  if constexpr (Symetric) symcont = false;
                }
                else if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
                {
                  Vec3d dr { rx_b[p_b] - rx_a[p_a] , ry_b[p_b] - ry_a[p_a] , rz_b[p_b] - rz_a[p_a] };
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
                      func( dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get( cell_a , p_a, p_nbh_index , nbh_data_ctx ) );
                    }
                  }
                  ++ p_nbh_index;
                }
              }
            }
          }

          if(Symetric && particle_offset==nullptr) { stream = chunknbh_stream_to_next_particle( stream , chunk , nchunks , cg , cell_groups ); }

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
    /*** end of chunk version implementation ***/
    
    /*************************************************
     *** Simple neighbors traversal implementation ***
     *************************************************/
    if constexpr ( NbhIteratorTraits<decltype(optional.nbh)>::is_simple )
    {
      const auto & cell_nbh = onika::cuda::vector_data( optional.nbh.m_cell_neighbors )[cell_a];
      const auto* __restrict__ nbh_start = onika::cuda::vector_data( cell_nbh.nbh_start );
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
          const auto * __restrict__ p_a_neighbors = neighbors + nbh_start[p_a];
          const unsigned int n_nbh = nbh_start[p_a+1] - nbh_start[p_a];
          for( unsigned int p_nbh_index=0 ; p_nbh_index<n_nbh ; p_nbh_index++ )
          {
            const auto [ cell_b , p_b ] = p_a_neighbors[p_nbh_index];
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
                func( dr, d2, cells[cell_a][cp_fields.get(onika::tuple_index_t<FieldIndex>{})][p_a] ... , cells , cell_b, p_b, optional.nbh_data.get( cell_a , p_a, p_nbh_index , nbh_data_ctx ) );
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

}

