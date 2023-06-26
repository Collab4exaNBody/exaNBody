#pragma once

#include <mutex>
#include <functional>
#include <iostream>
#include <ios>

#include <exanb/core/log.h>

namespace exanb
{

  class OperatorSlotResource
  {    
  public:
    
    OperatorSlotResource( void* memory_ptr = nullptr );
    OperatorSlotResource( std::function<void*()> allocator, std::function<void(void*)> deleter=nullptr );
    ~OperatorSlotResource();
    
    inline bool has_allocator() const { return ! (m_allocate==nullptr); }
    inline bool has_deleter() const { return ! (m_deleter==nullptr); }
    inline bool is_null() const { return m_memory_ptr==nullptr && m_allocate==nullptr; }
    
    inline void* memory_ptr() const { return m_memory_ptr; }
    void* check_allocate();
    void free();

  private:
    void* m_memory_ptr = nullptr;
    std::function<void*()> m_allocate = nullptr;
    std::function<void(void*)> m_deleter = nullptr;
    // std::vector< uint64_t > m_access_masks;
    // std::mutex m_mutex;
  };

  std::ostream& operator << ( std::ostream& out, const OperatorSlotResource& r );

}

