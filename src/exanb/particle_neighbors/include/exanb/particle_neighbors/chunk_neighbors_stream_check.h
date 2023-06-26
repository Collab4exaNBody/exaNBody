#pragma once

#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>

#include <onika/cuda/cuda_context.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>


namespace exanb
{

    template<class GridT, class ChunkSizeT>
    inline int64_t chunk_neighbors_stream_check(const GridT& grid, ChunkSizeT CS, size_t cell_a, const uint16_t* input_stream, ssize_t stream_size, std::vector<unsigned int>& cell_a_particle_chunk_count)
    {
      int64_t total_nbh = 0;
      
      const auto cells = grid.cells();
      const IJK dims = grid.dimension();
      
      const unsigned int cell_a_particles = cells[cell_a].size();
      const IJK loc_a = grid_index_to_ijk( dims , cell_a );
      const auto stream_info = chunknbh_stream_info( input_stream , cell_a_particles );
      const uint16_t* stream_base = stream_info.stream;
      const uint16_t* __restrict__ stream = stream_base;
      const uint32_t* __restrict__ particle_offset = stream_info.offset;
      const int32_t poffshift = stream_info.shift;

      cell_a_particle_chunk_count.assign( cell_a_particles , 0 );
      for(unsigned int p_a=0; p_a<cell_a_particles; p_a++ )
      {
        if( particle_offset!=nullptr )
        {
          if( stream != (stream_base + particle_offset[p_a] + poffshift) )
          {
            printf("nbh stream is inconsistent with offset table cell=%lu, p=%d\n",cell_a,p_a);
            ONIKA_CU_ABORT();
          }
          stream = stream_base + particle_offset[p_a] + poffshift;
        }
        const unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list        
        for(unsigned int cg=0; cg<cell_groups ;cg++)
        {
          assert( (stream-stream_base) < stream_size );
          const uint16_t cell_b_enc = *(stream++);
          if( cell_b_enc < GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE )
          {
            printf("nbh stream corrupted: cell=%llu, p=%u, cg=%u/%u, spos=%d\n",cell_a,p_a,cg,cell_groups,int(stream-input_stream) );
            ONIKA_CU_ABORT();
          }
          const IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
          if( !( loc_b.i>=0 && loc_b.j>=0 && loc_b.j>=0 && loc_b.i<dims.i && loc_b.j<dims.j && loc_b.j<dims.j ) )
          {
            printf("nbh stream corrupted: cell=%llu, p=%u, cg=%u/%u, loc=%d,%d,%d\n",cell_a,p_a,cg,cell_groups,int(loc_b.i),int(loc_b.j),int(loc_b.k));
            ONIKA_CU_ABORT();            
          }
          const size_t cell_b = grid_ijk_to_index( dims , loc_b );
          const unsigned int nbh_cell_particles = cells[cell_b].size();
          
          assert( (stream-stream_base) < stream_size );
          const unsigned int nchunks = *(stream++);
          for(unsigned int chunk=0;chunk<nchunks;chunk++)
          {
            ++ cell_a_particle_chunk_count[p_a];
            assert( (stream-stream_base) < stream_size );
            const unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
            for(unsigned int i=0;i<CS;i++)
            {
              const unsigned int p_b = chunk_start + i;
              if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
              {
                ++ total_nbh;
              }
            }
          }
        }
      }

      return total_nbh;
    }
    
}

