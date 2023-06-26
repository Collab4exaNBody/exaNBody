#pragma once

namespace exanb
{
  

  //template<class AllocatorT = onika::memory::CudaManagedAllocator<uint16_t> >
  struct GridChunkNeighborsHostWriteAccessor //Impl
  {
    onika::memory::MemoryPartionnerMT& m_fixed_stream_pool;
    uint16_t * * const m_cell_stream = nullptr ;
    uint32_t * const m_cell_stream_size = nullptr;
    const unsigned int m_chunk_size = 4;
    onika::memory::GenericHostAllocator m_alloc = {};

    inline GridChunkNeighborsHostWriteAccessor /*Impl*/( GridChunkNeighbors& cn )
      : m_fixed_stream_pool( cn.m_fixed_stream_pool )
      , m_cell_stream( cn.m_cell_stream.data() )
      , m_cell_stream_size( cn.m_cell_stream_size.data() )
      , m_chunk_size( cn.m_chunk_size )
      , m_alloc( cn.m_alloc )
    {}

    inline GridChunkNeighborsData data() const { return m_cell_stream; }
    inline const uint16_t * cell_stream(size_t i) const { return m_cell_stream[i]; }
    inline size_t cell_stream_size(size_t i) const { return m_cell_stream_size[i]; }
    inline uint16_t* allocate( size_t cell_i, size_t n )
    {
      if( n == 0 )
      {
        m_cell_stream[cell_i] = nullptr;
        m_cell_stream_size[cell_i] = 0 ;
        return nullptr;
      }
      // assert( cell_i < m_cell_stream.size() && cell_i < m_cell_stream_size.size() );      
      uint16_t* p = (uint16_t*) m_fixed_stream_pool.alloc( n * sizeof(uint16_t) , onika::memory::MINIMUM_CUDA_ALIGNMENT );
      if( p == nullptr )
      {
        p = (uint16_t*) m_alloc.allocate( n * sizeof(uint16_t) , onika::memory::MINIMUM_CUDA_ALIGNMENT );
        assert( ! m_fixed_stream_pool.contains(p) );
      }
      m_cell_stream[cell_i] = p;
      m_cell_stream_size[cell_i] = (p!=nullptr) ? ( n * sizeof(uint16_t) ) : 0 ;
      return p;
    }
  };

  // using GridChunkNeighborsHostWriteAccessor = GridChunkNeighborsHostWriteAccessorImpl<>;

}

