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

