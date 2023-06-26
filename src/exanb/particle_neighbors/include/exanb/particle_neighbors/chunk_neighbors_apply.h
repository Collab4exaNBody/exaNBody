#pragma once

#include <exanb/particle_neighbors/chunk_neighbors.h>

namespace exanb
{
  

  struct NullStreamHandlerFunc { inline void operator () (size_t,ptrdiff_t) const {} };

  template<class GridT, class ChunkSizeT, class FuncT, bool Symetric>
  static inline void apply_cell_particle_neighbors(GridT& grid, GridChunkNeighborsData nbh, size_t cell_a, IJK loc_a, ChunkSizeT CS, std::integral_constant<bool,Symetric> , FuncT func)
  {
    //const uint16_t* __restrict__ stream_start = nbh[cell_a].data();
    auto * cells = grid.cells();

    const unsigned int cell_a_particles = cells[cell_a].size();
    const auto stream_info = chunknbh_stream_info( nbh[cell_a] , cell_a_particles );
    const uint16_t* stream_base = stream_info.stream;
    const uint16_t* __restrict__ stream = stream_base;
    const uint32_t* __restrict__ particle_offset = stream_info.offset;
    const int32_t poffshift = stream_info.shift;
        
    const IJK dims = grid.dimension();
    
    for(unsigned int p_a=0;p_a<cell_a_particles;p_a++)
    {
      size_t p_nbh_index = 0;    
      if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;

      //sfunc( p_a , stream - stream_start );
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
              func( p_a, cell_b, p_b , p_nbh_index );
              ++ p_nbh_index; 
            }
          }
        }
      }

      if(Symetric && particle_offset==nullptr) { stream = chunknbh_stream_to_next_particle( stream , chunk , nchunks , cg , cell_groups ); }
    }

  }

  template<class GridT, typename FuncT, bool Symmetric=false>
  static inline void apply_cell_particle_neighbors(GridT& grid, const GridChunkNeighbors& nbh, size_t cell_a, IJK loc_a, std::integral_constant<bool,Symmetric> , FuncT func )
  {
    switch(nbh.m_chunk_size)
    {
      case 4 : apply_cell_particle_neighbors(grid, nbh.data(), cell_a, loc_a, std::integral_constant<unsigned int,4>() , std::integral_constant<bool,Symmetric>() , func ); break;
      case 8 : apply_cell_particle_neighbors(grid, nbh.data(), cell_a, loc_a, std::integral_constant<unsigned int,8>() , std::integral_constant<bool,Symmetric>() , func ); break;
      default: apply_cell_particle_neighbors(grid, nbh.data(), cell_a, loc_a, nbh.m_chunk_size                         , std::integral_constant<bool,Symmetric>() , func ); break;
    }
  }

}


