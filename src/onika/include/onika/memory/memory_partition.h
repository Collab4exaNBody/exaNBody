#pragma once

#include <cstdlib>
#include <iostream>
#include <iostream>
#include <type_traits>

namespace onika { namespace memory
{

  template<bool _ConcurrentAccess = false, size_t _AllocGranularityLog2 = 6>
	struct MemoryPartionnerImpl
	{
	  static inline constexpr bool ConcurrentAccess = _ConcurrentAccess;
	  static inline constexpr unsigned int AllocGranularityLog2 = _AllocGranularityLog2;
	  static inline constexpr size_t AllocGranularity = 1ull << AllocGranularityLog2;
	  static inline constexpr size_t AllocGuardSize = 1024ull * 1024ull * 1024ull; // 1Gb allocation size guard to prevent undetected capacity overflow in concurent access context
	
	  uint8_t* m_base_ptr = nullptr;
	  uint32_t m_capacity = 0;  // 32Bits only for GPU compatibility, this limits maximum  size of memory pool
	  uint32_t m_allocated = 0;
	  uint32_t m_overflow = 0;
	  
	  ONIKA_HOST_DEVICE_FUNC inline size_t max_capacity_bytes() const
	  {
	    return ( 1ull << (32+AllocGranularityLog2) ) - AllocGuardSize ;
	  }

	  ONIKA_HOST_DEVICE_FUNC inline size_t adjust_allocation_size(size_t sz)
	  {
	    size_t n_units = (sz+AllocGranularity-1) / AllocGranularity;
	    if( n_units*AllocGranularity > max_capacity_bytes() ) n_units = max_capacity_bytes() / AllocGranularity;
	    return n_units * AllocGranularity;
    }
	  
	  ONIKA_HOST_DEVICE_FUNC inline void set_allocated_memory(uint8_t* ptr, size_t sz)
	  {
	    m_base_ptr = ptr;
	    m_capacity = sz / AllocGranularity;
	    m_allocated = 0;
	    m_overflow = 0;
	  }
	  
	  ONIKA_HOST_DEVICE_FUNC inline void clear() { m_allocated = 0; }
	  
	  ONIKA_HOST_DEVICE_FUNC inline uint8_t* base_ptr() const { return m_base_ptr; }
	  
	  // return capacity in 'grain units, if AllocGranularity==8, return number of allocated bytes /8
	  ONIKA_HOST_DEVICE_FUNC inline size_t capacity() const { return m_capacity; }
	  
	  // returns allocated memory (capacity) in bytes, not in elemenatry 'grains'
	  ONIKA_HOST_DEVICE_FUNC inline size_t memory_bytes() const { return capacity() * AllocGranularity; }
	  
	  // returns available elmentary 'units' (AllocGranularity sized elements)
	  ONIKA_HOST_DEVICE_FUNC inline size_t available_units() const
	  {
	    if( !m_overflow && m_allocated < m_capacity ) return m_capacity - m_allocated;
	    else return 0;
	  }

	  ONIKA_HOST_DEVICE_FUNC inline size_t available_bytes() const
	  {
	    return available_units() * AllocGranularity;
	  }

    ONIKA_HOST_DEVICE_FUNC inline bool contains(const void* p) const
    {
      ptrdiff_t d = ( (const uint8_t*) p ) - m_base_ptr;
      return d >= 0 && static_cast<size_t>(d) < static_cast<size_t>(m_capacity)*AllocGranularity;
    }
	  
	  ONIKA_HOST_DEVICE_FUNC inline uint8_t* alloc(size_t n, size_t a = AllocGranularity )
	  {
	    if( n == 0 || n > AllocGuardSize ) return nullptr;

      size_t alloc_offset = 0;

      if constexpr ( ConcurrentAccess )
      {
        const size_t required_alloc_bytes = n + a - 1;
        const size_t required_alloc = ( required_alloc_bytes + AllocGranularity - 1 ) / AllocGranularity;
        
        if( ( m_allocated  + required_alloc ) > m_capacity ) { m_overflow = 1; }
        if( ! m_overflow ) { alloc_offset = ONIKA_CU_ATOMIC_ADD( m_allocated , required_alloc ); }
	      if( ( alloc_offset + required_alloc ) > m_capacity ) { m_overflow = 1; }
	      if( m_overflow ) return nullptr;
	      
	      alloc_offset *= AllocGranularity;
	      
        uint8_t* alloc_ptr = m_base_ptr + alloc_offset;
	      const ptrdiff_t cur_ptr_value = alloc_ptr - (uint8_t*)nullptr;
	      const size_t moda = cur_ptr_value % a;
        const size_t pad = ( moda > 0 ) ? ( a - moda ) : 0 ;
        assert( ( pad + n ) <= required_alloc_bytes );
        
        alloc_offset += pad;
      }

      if constexpr ( !ConcurrentAccess )
      {
        uint8_t* alloc_ptr = m_base_ptr + m_allocated * AllocGranularity;
	      const ptrdiff_t cur_ptr_value = alloc_ptr - (uint8_t*)nullptr;
	      const size_t moda = cur_ptr_value % a;
        const size_t pad = ( moda > 0 ) ? ( a - moda ) : 0 ;
        
        const size_t required_alloc_bytes = pad + n;
        const size_t required_alloc = ( required_alloc_bytes + AllocGranularity - 1 ) / AllocGranularity;
	      if( available_units() < required_alloc ) return nullptr;
	      
	      alloc_offset = ( m_allocated * AllocGranularity ) + pad;
        m_allocated += required_alloc;
	    }
	    
	    uint8_t* ptr = m_base_ptr + alloc_offset;
	    int modal = ( (int64_t)( ptr - (uint8_t*)nullptr ) ) % a;
      if( modal != 0 )
      {
        printf("Misaligned address in MemoryPartionnerImpl<%s>::alloc :\nptr=%p, a=%d, mod=%d\nbase=%p, offset=%d, alloc'd=%d\n",(ConcurrentAccess?"true":"false"),
               ptr,int(a), modal, m_base_ptr,int(alloc_offset),int(m_allocated) );
        ONIKA_CU_ABORT();
      }
	    return ptr;
	  }

    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline T* alloc_type(size_t n=1)
    {
      return (T*) alloc( n * sizeof(T) , alignof(T) );
    }
	  
	};

  using MemoryPartionner = MemoryPartionnerImpl<false>;
  using MemoryPartionnerMT = MemoryPartionnerImpl<true>;

}

}

