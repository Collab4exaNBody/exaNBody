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
#include <exanb/particle_neighbors/chunk_neighbors.h>

namespace exanb
{
   
  GridChunkNeighborsData GridChunkNeighbors::data() const
  {
    return m_cell_stream.data();
  }
  
  const uint16_t * GridChunkNeighbors::cell_stream(size_t i) const
  {
    return m_cell_stream[i];
  }
  
  size_t GridChunkNeighbors::cell_stream_size(size_t i) const
  {
    return m_cell_stream_size[i];
  }

  size_t GridChunkNeighbors::number_of_cells() const
  {
    assert( m_cell_stream.size() == m_cell_stream_size.size() );
    return m_cell_stream.size();
  }

  void GridChunkNeighbors::update_stream_pool_hint()
  {
    size_t n = m_cell_stream_size.size();
    assert( n == m_cell_stream.size() );
    m_nb_dyn_alloc = 0;
    m_stream_pool_hint = 0;
    for(size_t i=0;i<n;i++)
    {
      m_stream_pool_hint += m_cell_stream_size[i];
      if( m_cell_stream[i]!=nullptr && ! m_fixed_stream_pool.contains(m_cell_stream[i]) )
      {
        ++ m_nb_dyn_alloc;
      }
    }
  }
  
  void GridChunkNeighbors::set_number_of_cells(size_t n)
  {
    clear();
    m_cell_stream.assign( n , nullptr );
    m_cell_stream_size.assign( n , 0 );
  }
  
  void GridChunkNeighbors::set_chunk_size( unsigned int cs )
  {
    m_chunk_size = cs;
  }
}


