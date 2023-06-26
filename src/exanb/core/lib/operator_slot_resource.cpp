#include <exanb/core/operator_slot_resource.h>
#include <exanb/core/log.h>
#include <cassert>

namespace exanb
{

  OperatorSlotResource::OperatorSlotResource( void* memory_ptr )
    : m_memory_ptr( memory_ptr )
  {
  }

  OperatorSlotResource::OperatorSlotResource( std::function<void*()> allocator , std::function<void(void*)> deleter )
    : m_memory_ptr( nullptr )
    , m_allocate( allocator )
    , m_deleter( deleter )
  {
  }

  OperatorSlotResource::~OperatorSlotResource()
  {
    free();
  }

  void* OperatorSlotResource::check_allocate()
  {
#   pragma omp critical(OperatorSlotResource_check_allocate)
    if( m_memory_ptr==nullptr && m_allocate!=nullptr )
    {
      m_memory_ptr = m_allocate();
    }

    return m_memory_ptr;
  }

  void OperatorSlotResource::free()
  {
    if( m_memory_ptr != nullptr )
    {
      if( m_deleter == nullptr )
      {
        lerr << "resource @"<<m_memory_ptr<<" leaks because it lacks a deleter" << std::endl;
      }
      else
      {
        // ldbg << "deallocate resource @"<<m_memory_ptr<<std::endl;
#       pragma omp critical(OperatorSlotResource_free)
        m_deleter( m_memory_ptr ); 
      }
    }
    m_memory_ptr = nullptr;
  }

  std::ostream& operator << ( std::ostream& out, const OperatorSlotResource& r )
  {
    return out << "ptr="<<r.memory_ptr() /*<<",excl="<<r.is_exlusively_locked()*/ <<",has_alloc="<<r.has_allocator()<<",has_del="<<r.has_deleter();
  }

}


