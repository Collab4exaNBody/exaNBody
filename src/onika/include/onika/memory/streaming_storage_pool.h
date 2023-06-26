#pragma once

#include <onika/memory/streaming_storage_unit.h>
#include <atomic>
#include <memory>
#include <cstdint>
#include <thread>

namespace onika
{
  namespace memory
  {
    
    struct StreamingStoragePool
    {
      using storage_unit_t = StreamingStorageUnit;    
      static inline constexpr size_t NbStorageUnits = 64;
      static inline constexpr size_t AllocationRetryCount = NbStorageUnits * 2;
      storage_unit_t m_storage_units[NbStorageUnits];
      std::atomic<uint32_t> m_storage_unit_idx = 0;

      inline std::pair<size_t,size_t> memory_usage()
      {
        size_t allocd=0, freed=0;
        for(size_t i=0;i<NbStorageUnits;i++)
        {
          auto && [a,f] = m_storage_units[i].memory_usage();
          allocd += a;
          freed += f;
        }
        return { allocd , freed };
      }

      inline std::pair<char*,size_t> allocate(size_t sz)
      {
        size_t idx = 0;
        char* alloc_ptr = nullptr;
        size_t sidx = m_storage_unit_idx.load( std::memory_order_relaxed );
        size_t i=0;
        for(i=0;i<AllocationRetryCount && alloc_ptr==nullptr ;i++)
        {
          idx = ( sidx + i ) % NbStorageUnits;
          alloc_ptr = m_storage_units[idx].allocate( sz );
        }
        m_storage_unit_idx.store( idx , std::memory_order_relaxed );
        if(alloc_ptr!=nullptr && i>0) --i;
        return { alloc_ptr , i };
      }

      static inline void free( void* ptr, size_t sz )
      {
        storage_unit_t::free( ptr , sz );
      }

      struct allocate_nofail_ret { char* ptr; size_t retry; size_t yield; size_t ctx_sw; };

      inline allocate_nofail_ret allocate_nofail(size_t sz, bool taskyield_on_fail = true)
      {
        auto && [ ptr , retry ] = allocate(sz);
        size_t yield_count = 0;
        size_t ctx_sw = 0;
        while( ptr == nullptr )
        {
          if( taskyield_on_fail )
          {
            ++ yield_count;
            auto cur_tid = std::this_thread::get_id();
#           pragma omp taskyield
            if( cur_tid != std::this_thread::get_id() ) ++ ctx_sw;
          }
          auto && [ p , r ] = allocate(sz);
          ptr = p;
          retry += r;
        }
        return { ptr , retry , yield_count , ctx_sw };
      }

    };

  }
  
}

