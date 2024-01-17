#pragma once

#include <cstdlib>
#include <thread>
#include <atomic>
#include <mutex>
#include <omp.h>

#include <cassert>
#include <chrono>
#include <functional>

#include <onika/debug.h>
#include <onika/cuda/cuda_context.h>

namespace exanb
{
  // this is a hint, there is no guarantee
  static constexpr int max_threads_hint = std::max<int>( XSTAMP_ADVISED_HW_THREADS , 16 );

  //**************** spin lock implementation *********************
  static constexpr size_t ATOMIC_LOCK_ALIGNMENT=4;
  static constexpr size_t ATOMIC_LOCK_WAIT_NANOSEC=0;
  
  template<size_t NANOSECS=ATOMIC_LOCK_WAIT_NANOSEC, size_t ALIGN_BYTES=ATOMIC_LOCK_ALIGNMENT>
  class alignas(ALIGN_BYTES) atomic_lock_spin_mutex
  {
    public:
      ONIKA_HOST_DEVICE_FUNC inline bool try_lock()
      {
          //return !_lock.test_and_set(std::memory_order_acquire);
          return ONIKA_CU_ATOMIC_FLAG_TEST_AND_SET( _lock );
      }

      ONIKA_HOST_DEVICE_FUNC inline void lock()
      {
          size_t wait_time = NANOSECS;
          while (!try_lock())
          { 
              if(wait_time>0)
              {
                  //std::this_thread::sleep_for(std::chrono::nanoseconds(wait_time));
                  ONIKA_CU_NANOSLEEP( wait_time );
              }
              wait_time += NANOSECS;
          }
      }

      ONIKA_HOST_DEVICE_FUNC inline void unlock()
      {
          //_lock.clear(std::memory_order_release);
          ONIKA_CU_ATOMIC_FLAG_CLEAR( _lock );
      }

    private:
      //std::atomic_flag _lock = ATOMIC_FLAG_INIT;
      ::onika::cuda::onika_cu_atomic_flag_t _lock = ONIKA_CU_ATOMIC_FLAG_INIT;
  };

  struct null_spin_mutex
  {
      uint8_t m_xxx;
      inline bool try_lock() { return true; }
      inline void lock() { }
      inline void unlock() { }
  };

  class omp_spin_mutex
  {
  private:
    omp_lock_t m_lock;
  public: 
    inline void init() { omp_init_lock(&m_lock); }
    inline omp_spin_mutex() { init(); }
    inline ~omp_spin_mutex() { omp_destroy_lock(&m_lock); }
    inline void lock() { omp_set_lock(&m_lock); }
    inline void unlock() { omp_unset_lock(&m_lock); }
    inline bool try_lock() { return omp_test_lock(&m_lock); }
  };

  class stl_spin_mutex
  {
  private:
    std::mutex m_mutex;
  public: 
    inline void lock    () { m_mutex.lock(); }
    inline void unlock  () { m_mutex.unlock(); }
    inline bool try_lock() { return m_mutex.try_lock(); }
  };


  using spin_mutex = atomic_lock_spin_mutex<>;
//  using spin_mutex = omp_spin_mutex;
  
  class spin_mutex_array
  {
  public:
    spin_mutex_array() = default;
    inline spin_mutex_array( size_t n ) { resize(n); }
    inline spin_mutex_array( const spin_mutex_array & other )
    {
      // for tests only, should be a deleted function, as the state of locks cannot be copied
      resize( other.m_size );
    }
    inline spin_mutex_array& operator = ( const spin_mutex_array & other )
    {
      // for tests only, should be a deleted function, as the state of locks cannot be copied
      resize( other.m_size );
      return *this;
    }
 
    inline spin_mutex_array( spin_mutex_array && other )
    {
      m_size = other.m_size;
      m_array = other.m_array;
      other.m_size = 0;
      other.m_array = 0;
    }
    inline spin_mutex_array& operator = ( spin_mutex_array && other )
    {
      m_size = other.m_size;
      m_array = other.m_array;
      other.m_size = 0;
      other.m_array = 0;
      return *this;
    }

    inline ~spin_mutex_array() { resize(0); }
  
    ONIKA_HOST_DEVICE_FUNC inline spin_mutex& operator [] (size_t i) { return m_array[i]; }
    
    inline void resize(size_t n)
    {
      if(n==m_size) return;
      if(m_array!=nullptr && m_size>0) { delete [] m_array; m_array = nullptr; }
      assert( m_array == nullptr );
      m_size = n;
      if(m_size>0) { m_array = new spin_mutex[m_size]; }
    }
    inline spin_mutex* data() { return m_array; }
    
    inline size_t size() const { return m_size; }
    
    inline size_t memory_bytes() const { return sizeof(spin_mutex_array) + size()*sizeof(spin_mutex); }
 
   private:
    spin_mutex* m_array = nullptr;
    size_t m_size = 0;
  };
  //using spin_mutex = stl_spin_mutex;   

  // simple alias for array of lock arrays
  using GridParticleLocks = std::vector<exanb::spin_mutex_array>;

  // =========== light weight shared mutex =====================
  // only capable of try_lock and try_lock_shared.
  // uses a single 32-bits atomic
  // sizeof(LightWeightSharedMutext) = 4
  // sizeof(std::shared_mutex) = 56
  struct LightWeightSharedMutex
  {
    static constexpr uint32_t EXCLUSIVE_LOCK = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t MAX_SHARE_COUNT = EXCLUSIVE_LOCK - 1;
    std::atomic<uint32_t> m_counter = 0;
    inline bool try_lock_shared()
    {
      uint32_t current = m_counter.load( std::memory_order_relaxed );
      if( current < MAX_SHARE_COUNT )
      {
        return m_counter.compare_exchange_weak( current, current+1, std::memory_order_acquire, std::memory_order_relaxed );
      }
      else { return false; }
    }
    inline void unlock_shared()
    {
      ONIKA_DEBUG_ONLY( uint32_t prev = ) m_counter.fetch_sub(1 , std::memory_order_relaxed );
      assert( prev!=0 && prev!=EXCLUSIVE_LOCK );
    }
    inline bool try_lock()
    {
      uint32_t expected = 0;
      return m_counter.compare_exchange_weak( expected, EXCLUSIVE_LOCK, std::memory_order_acquire, std::memory_order_relaxed );
    }
    inline void unlock()
    {
      ONIKA_DEBUG_ONLY( uint32_t prev = ) m_counter.exchange( 0 , std::memory_order_release );
      assert( prev == EXCLUSIVE_LOCK );
    }
  };


  //**************** reliable thread indexing *********************
  size_t get_thread_index();


  //**************** per thread timing functions *********************
  std::chrono::nanoseconds wall_clock_time();
  std::chrono::nanoseconds get_thread_cpu_time();
  
}


