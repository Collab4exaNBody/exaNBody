#pragma once

#include <utility>
#include <atomic>
#include <cstdlib>
#include <cassert>
#include <iostream>

#define ONIKA_POINTER_FIFO_PROFILE 1

namespace onika
{
  namespace memory
  {
  
    template<class T, size_t _MaxBufferSize = 1024*1024>
    class PointerFifo
    {
    public:   
      static constexpr size_t MaxBufferSize = _MaxBufferSize;

      inline PointerFifo()
      {
        for(size_t i=0;i<MaxBufferSize;i++)
        {
          m_buffer[i].store( nullptr , std::memory_order_relaxed );
        }
        std::atomic_thread_fence( std::memory_order_release );
      }
      
      // =================== FIFO push task ================
      // push one pointer into FIFO
      // ===================================================
      inline void push_back( T* item )
      {
        assert( item != nullptr );
        bool pushed = false;
        
#       ifdef ONIKA_POINTER_FIFO_PROFILE
        size_t n_retry = 0;
#       endif

        int32_t qend = m_buffer_end.load( std::memory_order_consume );
        
        while( ! pushed )
        {
          if( qend >= 0 ) // not spin locked
          {
            int32_t tmp_qend = qend % MaxBufferSize;
            uint32_t idx = tmp_qend;
            tmp_qend = - ( tmp_qend + 1 );
            bool spin_locked = m_buffer_end.compare_exchange_weak( qend , tmp_qend , std::memory_order_acq_rel , std::memory_order_consume );
            if( spin_locked )
            {
              m_buffer[idx].store( item , std::memory_order_relaxed );
              m_buffer_end.store( (qend+1)% MaxBufferSize , std::memory_order_release );
              pushed = true;
            }
#           ifdef ONIKA_POINTER_FIFO_PROFILE
            else { ++ n_retry; }
#           endif
          }
          else
          {
            qend = m_buffer_end.load( std::memory_order_consume );
#           ifdef ONIKA_POINTER_FIFO_PROFILE
            ++ n_retry;
#           endif
          }
        }

#       ifdef ONIKA_POINTER_FIFO_PROFILE
        // stats update
        m_push_retry.fetch_add(n_retry,std::memory_order_relaxed);
        m_total_pushed.fetch_add(1,std::memory_order_relaxed);
#       endif
      }


      // =================== FIFO pop task ================
      // take one available task from FIFO
      // ==================================================
      inline std::pair<T*,size_t> pop_front()
      {
        T* item = nullptr;

#       ifdef ONIKA_POINTER_FIFO_PROFILE
        size_t n_retry = 0;
#       endif
    
        int32_t qend = m_buffer_end.load( std::memory_order_acquire ); // acquire ensures that task data write happens-before
        if( qend < 0 ) qend = - (qend + 1 );

        int32_t qstart = m_buffer_start.load( std::memory_order_consume );

        while( item == nullptr )
        {
          if( qstart >= 0 ) // not spin locked
          {
            int32_t tmp_qstart = qstart % MaxBufferSize;
            if( tmp_qstart == int32_t( qend % MaxBufferSize ) ) // the task queue is empty
            {
              // stats update
#             ifdef ONIKA_POINTER_FIFO_PROFILE
              m_pop_retry.fetch_add(n_retry,std::memory_order_relaxed);
#             endif
              return { nullptr , 0 };
            }
            uint32_t idx = tmp_qstart;
            tmp_qstart = - ( tmp_qstart + 1 );
            bool spin_locked = m_buffer_start.compare_exchange_weak( qstart , tmp_qstart , std::memory_order_acq_rel , std::memory_order_consume );
            if( spin_locked )
            {
              item = m_buffer[idx].load( std::memory_order_relaxed );
              assert( item != nullptr );
              m_buffer[idx].store( nullptr , std::memory_order_relaxed );
              m_buffer_start.store( (qstart+1)% MaxBufferSize , std::memory_order_release );
            }
#           ifdef ONIKA_POINTER_FIFO_PROFILE
            else { ++ n_retry; }
#           endif
          }
          else
          {
            qstart = m_buffer_start.load( std::memory_order_consume );
#           ifdef ONIKA_POINTER_FIFO_PROFILE
            ++ n_retry;
#           endif
          }
        }

        // stats update
#       ifdef ONIKA_POINTER_FIFO_PROFILE
        m_pop_retry.fetch_add(n_retry,std::memory_order_relaxed);
#       endif
        
        // including task returned. a return of { task , 1 } indicates that the returned task *may* be the last available
        size_t estimated_available_items = ( ( qend + 2 * MaxBufferSize ) - qstart ) % MaxBufferSize ;
        return { item , estimated_available_items };
      }

      // unsafe, must be mutex guarded and enclosed with memory fences
      inline std::tuple< std::atomic<T*>* , int32_t , int32_t > data() const
      {
        int32_t qstart = m_buffer_start.load( std::memory_order_relaxed );
        qstart = qstart % MaxBufferSize;
        int32_t qend = m_buffer_end.load( std::memory_order_relaxed ); // acquire ensures that task data write happens-before
        if( qend < 0 ) qend = - (qend + 1 );
        qend = qend % MaxBufferSize;
        return { m_buffer , qstart , qend };
      }

      // unsafe, must be mutex guarded and enclosed with memory fences
      // caller must guarantee that all pointers have been set to nullptr before calling this
      inline void clear()
      {
        m_buffer_start.store( 0 , std::memory_order_relaxed );
        m_buffer_end.store( 0 , std::memory_order_relaxed );
      }

      // unasafe, approximated size is returned
      inline size_t size() const
      {
        return ( 2*MaxBufferSize + m_buffer_end.load() - m_buffer_start.load() ) % MaxBufferSize;
      }

      // statistics about lock free strategy
      inline void print_stats()
      {
        std::atomic_thread_fence( std::memory_order_acquire );
        std::cout << "FIFO size           = "<< size() << std::endl;
#       ifdef ONIKA_POINTER_FIFO_PROFILE
        std::cout << "total pushed        = "<<m_total_pushed.load()<<std::endl;
        std::cout << "task push retry     = "<<m_push_retry.load()<<std::endl;
        std::cout << "task pop retry      = "<<m_pop_retry.load()<<std::endl;
#       endif
        size_t n_non_null = 0;
        for(size_t i=0;i<MaxBufferSize;i++)
        {
          if( m_buffer[i].load( std::memory_order_relaxed ) != nullptr ) ++ n_non_null;
        }
        std::cout << "non null pointers   = "<<n_non_null<<std::endl;
      }

    private:    
      std::atomic<int32_t> m_buffer_start = 0;
      std::atomic<int32_t> m_buffer_end = 0;
      std::atomic<T*> m_buffer[MaxBufferSize];
      
      // statistics
#     ifdef ONIKA_POINTER_FIFO_PROFILE
      std::atomic<uint32_t> m_total_pushed = 0;
      std::atomic<uint32_t> m_push_retry = 0;
      std::atomic<uint32_t> m_pop_retry = 0;
#     endif
    };

  }
}

