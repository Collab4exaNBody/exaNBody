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

#pragma once

#include <mutex>
#include <functional>
#include <iostream>
#include <ios>

#include <onika/log.h>
#include <onika/type_utils.h>

namespace onika { namespace scg
{

  class OperatorSlotResource
  {
  public:
    OperatorSlotResource( const OperatorSlotResource& other ) = default;
    OperatorSlotResource( OperatorSlotResource && other );
    OperatorSlotResource& operator = ( OperatorSlotResource && other );

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
    std::function<void*()> m_allocate = nullptr;
    std::function<void(void*)> m_deleter = nullptr;
    void* m_memory_ptr = nullptr;
  };

  std::ostream& operator << ( std::ostream& out, const OperatorSlotResource& r );


  // default constructor/destructor allocators and deleters
  template<class T,bool=std::is_destructible<T>::value> struct DefaultResourceDeleter
  {
    static inline std::function<void(void*)> build()
    {
      std::function<void(void*)> deleter = [](void* p) { delete reinterpret_cast<T*>(p); };
      return deleter;
    }
  };
  template<class T> struct DefaultResourceDeleter<T,false>
  {
    static inline std::function<void(void*)> build() { return nullptr; }
  };

  // utility resource allocator functions  
  template<class T>
  static inline std::shared_ptr<OperatorSlotResource> default_value_copy_constructor_resource(const T& defval)
  {
    return std::make_shared<OperatorSlotResource>( [defval]() -> void* { return new T(defval); } , DefaultResourceDeleter<T>::build() );
  }

  template<class T>
  static inline std::shared_ptr<OperatorSlotResource> default_constructor_resource( TypePlaceHolder<T> = {} )
  {
    return std::make_shared<OperatorSlotResource>( []() -> void* { return new T(); } , DefaultResourceDeleter<T>::build() );
  }

  template<class T>
  static inline std::shared_ptr<OperatorSlotResource> borrowed_pointer_resource( T* ptr )
  {
    return std::make_shared<OperatorSlotResource>( [ptr]() -> void* { return ptr; } );
  }


} }

