#pragma once

namespace exanb
{
  

  struct ChunkNeighborVariableCapacityTemp
  {
    inline ChunkNeighborVariableCapacityTemp(ChunkNeighborsPerThreadScratchEncoding& scratch, size_t n_particles_a)
    : cell_a_particle_nbh( scratch.cell_a_particle_nbh ) 
    , p_a_nbh_status( scratch.encoding_status )
    {
      cell_a_particle_nbh.resize( n_particles_a );
      for(unsigned int i=0; i<n_particles_a; i++)
      {
        cell_a_particle_nbh[i].assign( 1 , 0 ); // first cell group's chunk counter allocated and set to 0
      }
    }
    
    inline void begin_sub_cell( unsigned int p_start_a, unsigned int p_end_a )
    {
      assert( p_end_a >= p_start_a );
      p_a_nbh_status.assign( p_end_a-p_start_a , { { 0 , 0 } , std::numeric_limits<unsigned int>::max() } );
    }
    
    static inline constexpr unsigned int avail_stream_space( unsigned int )
    {
      return 1024*1024; // there's always free space, while capacity can grow
    }

    inline void end_sub_cell( unsigned int, unsigned int)
    {
    }

    std::vector< std::vector< uint16_t > > & cell_a_particle_nbh;
    std::vector< ChunkNeighborsEncodingStatus > & p_a_nbh_status;
  };

}

