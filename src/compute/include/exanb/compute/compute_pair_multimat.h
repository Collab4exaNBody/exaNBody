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

#include <exanb/core/grid.h>
#include <exanb/compute/compute_pair_buffer.h>
#include <exanb/compute/compute_pair_optional_args.h>
#include <exanb/compute/compute_pair_function.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/thread.h>
#include <exanb/core/particle_type_pair.h>
//#include "exanb/debug/debug_particle_id.h"

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_pointer_tuple.h>

#include <vector>
#include <functional>
#include <memory>
#include <string>

namespace exanb
{

  struct IdentityPairIdMap
  {
    inline constexpr int operator [] (int i) { return i; }
  };

  template< class GridT,
            class ComputePairBufferT, 
            class OptionalArgsT, 
            class... field_ids>
  static inline void compute_pair_multimat_cell(
    GridT& grid,
    ComputePairBufferT* tab,
    size_t cell_a,
    const std::vector< double >& rcuts,
    const std::vector< std::shared_ptr< ComputePairOperator< FieldSet<field_ids...> > > >& funcs,
    const unsigned int NTYPES,
    const int * __restrict__ pair_id_map,
    const OptionalArgsT& optional, // locks are needed if symmetric computation is enabled
    FieldSet< field_ids... >
    )
  {
    static constexpr auto const_1 = std::integral_constant<unsigned int,1>();
    static constexpr auto const_2 = std::integral_constant<unsigned int,2>();
    static constexpr auto const_4 = std::integral_constant<unsigned int,4>();
    static constexpr auto const_8 = std::integral_constant<unsigned int,8>();
    const unsigned int chunk_size = optional.nbh.m_chunk_size;
 
    if( pair_id_map != nullptr )
    {
      switch( chunk_size )
      {
        case 1 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,pair_id_map,optional, const_1, FieldSet<field_ids...>{}); break;
        case 2 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,pair_id_map,optional, const_2, FieldSet<field_ids...>{}); break;
        case 4 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,pair_id_map,optional, const_4, FieldSet<field_ids...>{}); break;
        case 8 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,pair_id_map,optional, const_8, FieldSet<field_ids...>{}); break;
        default:
          lerr << ":compute_pair_multimat_cell: chunk size "<<chunk_size<<" not supported. Accepted values are 1, 2, 4, 8." << std::endl;
          std::abort();
          break;
      }
    }
    else
    {
      switch( chunk_size )
      {
        case 1 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,IdentityPairIdMap{},optional, const_1, FieldSet<field_ids...>{}); break;
        case 2 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,IdentityPairIdMap{},optional, const_2, FieldSet<field_ids...>{}); break;
        case 4 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,IdentityPairIdMap{},optional, const_4, FieldSet<field_ids...>{}); break;
        case 8 : compute_pair_multimat_cell(grid,tab,cell_a,rcuts,funcs,NTYPES,IdentityPairIdMap{},optional, const_8, FieldSet<field_ids...>{}); break;
        default:
          lerr << "compute_pair_multimat_cell: chunk size "<<chunk_size<<" not supported. Accepted values are 1, 2, 4, 8." << std::endl;
          std::abort();
          break;
      }
    }
  }

  template< class GridT,
            class CST,
            class ComputePairBufferT,
            class OptionalArgsT,
            class PairIdMapT,
            class... field_ids>
  static inline void compute_pair_multimat_cell(
    GridT& grid,
    ComputePairBufferT* tab,
    size_t cell_a,
    const std::vector< double >& rcuts,
    const std::vector< std::shared_ptr< ComputePairOperator< FieldSet<field_ids...> > > >& funcs,
    const unsigned int NTYPES,
    PairIdMapT pair_id_map,
    const OptionalArgsT& optional, // locks are needed if symmetric computation is enabled
    CST CS,
    FieldSet< field_ids... >
    )
  {
    using exanb::chunknbh_stream_to_next_particle;
    using exanb::chunknbh_stream_info;
    using exanb::decode_cell_index;

    
    using BaseFields = FieldSet<field::_rx,field::_ry,field::_rz,field::_type>;
    using CellAFields = MergeFieldSet< BaseFields , FieldSet<field_ids...> >;
    using CellAPointerTuple = GridFieldSetPointerTuple<GridT,CellAFields>;
    using CellBPointerTuple = GridFieldSetPointerTuple<GridT,BaseFields>;
    static constexpr bool Symetric = false;
    // const unsigned int N_PAIRS = unique_pair_count(NTYPES);
    const unsigned int n_reduced_pairs = rcuts.size();

    assert( rcuts.size() == funcs.size() );
    assert( rcuts.size() <= unique_pair_count(NTYPES) );
//    assert( funcs.size() <= N_PAIRS );

    auto cells = grid.cells();
    IJK dims = grid.dimension();
    IJK loc_a = grid_index_to_ijk( dims, cell_a );

    const size_t cell_a_particles = cells[cell_a].size(); 
    CellAPointerTuple cell_a_pointers;
    cells[cell_a].capture_pointers(cell_a_pointers);
    // auto& cell_a_locks = optional.locks[cell_a];
    
    auto nbh_data_ctx = optional.nbh_data.make_ctx();

    const double*  __restrict__ rx_a       = cell_a_pointers[field::rx]; ONIKA_ASSUME_ALIGNED(rx_a);
    const double*  __restrict__ ry_a       = cell_a_pointers[field::ry]; ONIKA_ASSUME_ALIGNED(ry_a);
    const double*  __restrict__ rz_a       = cell_a_pointers[field::rz]; ONIKA_ASSUME_ALIGNED(rz_a);
    const uint8_t* __restrict__ type_a_ptr = cell_a_pointers[field::type]; ONIKA_ASSUME_ALIGNED(type_a_ptr);
 
    for(unsigned int i=0;i<n_reduced_pairs;i++)
    {
      tab[i].cell = cell_a;
      tab[i].count = 0;
    }

    // shift stream position if needed
    const auto stream_info = chunknbh_stream_info( optional.nbh.m_nbh_streams[cell_a] , cell_a_particles );
    const uint16_t* stream_base = stream_info.stream;
    const uint16_t* __restrict__ stream = stream_base;
    // const uint32_t* __restrict__ particle_offset = stream_info.offset;
    // const int32_t poffshift = stream_info.shift;
 
    for(size_t p_a=0;p_a<cell_a_particles;p_a++)
    {
      // initialize neighbor list traversal
      size_t p_nbh_index = 0;

      const double* __restrict__ rx_b = nullptr; ONIKA_ASSUME_ALIGNED(rx_b);
      const double* __restrict__ ry_b = nullptr; ONIKA_ASSUME_ALIGNED(ry_b);
      const double* __restrict__ rz_b = nullptr; ONIKA_ASSUME_ALIGNED(rz_b);
      const uint8_t* __restrict__ type_b_ptr = nullptr; ONIKA_ASSUME_ALIGNED(type_b_ptr);

      const unsigned int type_a = type_a_ptr[p_a];
      for(unsigned int i=0;i<n_reduced_pairs;i++)
      {
        tab[i].part = p_a;
        tab[i].ta = type_a;
      }

      unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list
      size_t cell_b = cell_a;
      unsigned int chunk = 0;
      unsigned int nchunks = 0;
      unsigned int cg = 0; // cell group index.
      bool symcont = true;

      for(cg=0; cg<cell_groups && symcont ;cg++)
      { 
        uint16_t cell_b_enc = *(stream++);
        IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
        cell_b = grid_ijk_to_index( dims , loc_b );
        unsigned int nbh_cell_particles = cells[cell_b].size();
        CellBPointerTuple cell_b_pointers;
        cells[cell_b].capture_pointers(cell_b_pointers);
        rx_b       = cell_b_pointers[field::rx]; ONIKA_ASSUME_ALIGNED(rx_b);
        ry_b       = cell_b_pointers[field::ry]; ONIKA_ASSUME_ALIGNED(ry_b);
        rz_b       = cell_b_pointers[field::rz]; ONIKA_ASSUME_ALIGNED(rz_b);
        type_b_ptr = cell_b_pointers[field::type]; ONIKA_ASSUME_ALIGNED(type_b_ptr);

        nchunks = *(stream++);
        for(chunk=0;chunk<nchunks && symcont;chunk++)
        {
          unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
          for(unsigned int i=0;i<CS && symcont;i++)
          {
            unsigned int p_b = chunk_start + i;
            if( Symetric && ( cell_b>cell_a || ( cell_b==cell_a && p_b>=p_a ) ) )
            {
              symcont = false;
            }
            else if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
            {
              unsigned int type_b = type_b_ptr[p_b];
              assert( type_b < NTYPES );
              
              Vec3d dr { rx_b[p_b] - rx_a[p_a] , ry_b[p_b] - ry_a[p_a] , rz_b[p_b] - rz_a[p_a] };
              dr = optional.xform.transformCoord( dr );
              double d2 = norm2(dr);
              
              unsigned int pair_ab_id = unique_pair_id(type_a,type_b);
              assert( pair_ab_id < unique_pair_count(NTYPES) );

              double rcut = rcuts[pair_ab_id];
              double rcut2 = rcut*rcut;

              const auto w = optional.nbh_data.get(cell_a,p_a,p_nbh_index,nbh_data_ctx);
              if( d2 <= rcut2 && w )
              {
                tab[pair_ab_id].ta = type_a;
                tab[pair_ab_id].tb = type_b;
                tab[pair_ab_id].process_neighbor(tab[pair_ab_id], dr, d2, cells, cell_b, p_b, w );
              }
              ++ p_nbh_index;
            }
          } 
        }
      } // end of loop on cell groups (end of traversal for p_a neighbors)
      
      // call compute function of different type pairs
      for(unsigned int pair_i=0;pair_i<n_reduced_pairs;pair_i++)
      {
        if( tab[pair_i].count > 0 )
        {
          (*(funcs[pair_i])) ( tab[pair_i], cell_a_pointers[onika::soatl::FieldId<field_ids>()][p_a] ... );
          tab[pair_i].count = 0;
        }
      }
    } // end of loop on cell_a particles


  } // end of compute_pair_multimat_cell


  template< typename GridT,
            typename OptionalArgsT,
            typename ComputePairBufferFactoryT,
            typename... field_ids>
  static inline void compute_pair_multimat(
    GridT& grid,
    const std::vector< double >& rcuts,
    const std::vector< std::shared_ptr< ComputePairOperator< FieldSet<field_ids...> > > >& funcs,
    const unsigned int NTYPES,
    const std::vector<int>& pair_id_map,
    bool enable_ghosts,
    OptionalArgsT optional_in, // locks are needed if symmetric computation is enabled
    ComputePairBufferFactoryT cpbuf_factory,
    FieldSet< field_ids... >
    )
  {
    using ComputeBuffer = typename ComputePairBufferFactoryT::ComputePairBuffer;
  
    //const unsigned int N_PAIRS = unique_pair_count(NTYPES);
//    const double rcut2 = rcut * rcut;
    const IJK grid_dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = grid_dims - (2*gl);

    assert( rcuts.size() == funcs.size() && rcuts.size() == unique_pair_count(NTYPES) );

    ldbg << "compute_pair_multimat : rcuts.size()=" << rcuts.size() << ", funcs.size()="<<funcs.size()<<", pair_id_map.size()="<<pair_id_map.size()<<", unique_pair_count(NTYPES)="<<unique_pair_count(NTYPES) <<  std::endl << std::flush;
    
    const int * idmapptr = nullptr;
    //if( !pair_id_map.empty() ) idmapptr = pair_id_map.data();

    unsigned int nb_reduced_pairs = rcuts.size();
    
#   pragma omp parallel
    {
      // create per tread local objects
      //auto nbh_it = optional.nbh;
      OptionalArgsT optional = optional_in;
      std::unique_ptr<ComputeBuffer[]> tab = std::make_unique<ComputeBuffer[]>( nb_reduced_pairs );
      //ComputeBuffer tab[ nb_reduced_pairs ];

      for(unsigned int i=0;i<nb_reduced_pairs;i++)
      {
        //std::cout << tab[i] <<  std::endl << std::flush;
        cpbuf_factory.init(tab[i]);
        tab[i].ta = particle_id_codec::MAX_PARTICLE_TYPE;
        tab[i].tb = particle_id_codec::MAX_PARTICLE_TYPE;
      }

      GRID_OMP_FOR_BEGIN(block_dims,_,block_cell_a_loc, schedule(dynamic) )
      {
        IJK cell_a_loc = block_cell_a_loc + gl;
        size_t cell_a = grid_ijk_to_index( grid_dims , cell_a_loc );
        compute_pair_multimat_cell(grid,tab.get(),cell_a,rcuts,funcs,NTYPES,idmapptr,optional, FieldSet< field_ids... >() );
      }
      GRID_OMP_FOR_END

    }

  }

}

