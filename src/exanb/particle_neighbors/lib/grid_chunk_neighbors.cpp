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


