#pragma once

#include <atomic>
#include <cstdint>
#include <cassert>

namespace onika
{

  namespace memory
  {

    // fast lock-free allocation/deallocation buffer unit.
    // may fail spuriously (as if it was full, even if it's not)
    // opportunistic deallocation (only when all allocated chunks happen to be freed)

    template<size_t _BufferSize=256*1024, size_t _Alignment=16>
    struct alignas(_Alignment) StreamingStorageUnitT
    {
      static constexpr size_t BufferSize = _BufferSize;
      static constexpr size_t Alignment = _Alignment;

      // two concatenated uint32_t, first is number of allocated bytes, second is number of free'd bytes
      static_assert( BufferSize < (1ull<<32) , "buffer size must not exceed 4Gb" );

      // memory bytes
      char m_buffer[BufferSize];
      
      // allocated and free bytes counters packed into a single 64 bits word
      std::atomic<uint64_t> m_cursor_pair = 0;

      inline std::pair<size_t,size_t> memory_usage()
      {
        uint64_t p = m_cursor_pair.load();
        uint32_t alloc_count = p >> 32;
        uint32_t free_count = p; // upper 32-bits vanish, remaining bits (lower 32 bits) are free count
        return { alloc_count , free_count };
      }

      static inline size_t allocation_size(uint64_t payload_sz)
      {
        const size_t sz = ( payload_sz+sizeof(StreamingStorageUnitT*) + Alignment-1 ) & (~(Alignment-1));
        return sz;
      }

      /* Attempts to allocate sz bytes.
         Note: may fail spuriously, and then return nullptr even if there is enough space
         will fail and return nullptr anyway if there is not enough space.
         deallocated bytes may be actually freed opportunisticly if freed bytes equal alloc'd bytes
      */
      inline char* allocate(uint64_t payload_sz)
      {
        const size_t sz = allocation_size( payload_sz );
        assert( sz < BufferSize );

        uint64_t p = m_cursor_pair.load( std::memory_order_relaxed );
        uint32_t alloc_count = p >> 32;
        uint32_t free_count = p; // upper 32-bits vanish, remaining bits (lower 32 bits) are free count
        if( alloc_count+sz > BufferSize )
        {
          if( free_count == alloc_count )
          {
            // are we lucky enough so that cursor hasn't changed since we looked at it ?
            bool alloc_ok = m_cursor_pair.compare_exchange_weak( p , sz<<32 , std::memory_order_relaxed , std::memory_order_relaxed );
            // if so, replace its value with a reset state + allocation os sz bytes
            if( alloc_ok )
            {
              * reinterpret_cast<StreamingStorageUnitT**>(m_buffer+payload_sz) = this;
              return m_buffer;
            }
          }
          return nullptr;
        }
        else
        {
          bool alloc_ok = m_cursor_pair.compare_exchange_weak( p , p + (sz<<32) , std::memory_order_relaxed , std::memory_order_relaxed );
          if( alloc_ok )
          {
            * reinterpret_cast<StreamingStorageUnitT**>(m_buffer+alloc_count+payload_sz) = this;
            return m_buffer+alloc_count;
          }
        }
        return nullptr;
      }
      
      /* guaranted to account for deallocation of sz bytes (no spurious failure) */
      inline void free_bytes( uint64_t sz )
      {
        m_cursor_pair.fetch_add( sz , std::memory_order_relaxed );
      }

      // if ptr is a pointer returned by the allocation of payload_sz bytes using a StreamingStorageUnitT instance,
      // then it returns the StreamingStorageUnitT instance containing ptr.
      // is payload_sz is not equal to original allocation size or ptr has not been allocated with StreamingStorageUnitT behavior is undefined
      static inline StreamingStorageUnitT* pointer_storage_unit(void* ptr, uint64_t payload_sz)
      {
        return * reinterpret_cast<StreamingStorageUnitT**>( reinterpret_cast<char*>(ptr) + payload_sz );
      }
            
      static inline void free( void* ptr , uint64_t payload_sz )
      {
        const size_t sz = allocation_size(payload_sz);
        pointer_storage_unit(ptr,payload_sz)->free_bytes( /*ptr ,*/ sz );
      }

    };

    using StreamingStorageUnit = StreamingStorageUnitT<>;

  } // namespace memory

} // namespace onika
