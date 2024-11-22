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
#include <onika/scg/operator_slot_resource.h>
#include <onika/log.h>
#include <cassert>

namespace onika { namespace scg
{

  OperatorSlotResource::OperatorSlotResource( void* memory_ptr )
    : m_memory_ptr( memory_ptr )
  {
  }

  OperatorSlotResource::OperatorSlotResource( std::function<void*()> allocator , std::function<void(void*)> deleter )
    : m_allocate  ( allocator )
    , m_deleter   ( deleter )
    , m_memory_ptr( nullptr )
  {
  }

  OperatorSlotResource::OperatorSlotResource( OperatorSlotResource && other )
  : m_allocate  ( std::move(other.m_allocate) )
  , m_deleter   ( std::move(other.m_deleter) )
  , m_memory_ptr( std::move(other.m_memory_ptr) )
  {
    other.m_allocate   = nullptr;
    other.m_deleter    = nullptr;
    other.m_memory_ptr = nullptr;
  }
  
  OperatorSlotResource& OperatorSlotResource::operator = ( OperatorSlotResource && other )
  {
    m_allocate   = std::move(other.m_allocate);
    m_deleter    = std::move(other.m_deleter);
    m_memory_ptr = std::move(other.m_memory_ptr);
    other.m_allocate   = nullptr;
    other.m_deleter    = nullptr;
    other.m_memory_ptr = nullptr;
    return *this;
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

} }


