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

#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>

#include <exanb/core/grid_fields.h>

#include <cstdlib>
#include <vector>
#include <type_traits>
#include <memory>

namespace exanb
{

  class FlatArrayDescriptor
  {
  public:
    virtual const char* name() const =0;
    virtual const char* short_name() const =0;
    virtual std::string type_name() const =0;
    virtual size_t size() const =0;
    virtual void resize(size_t sz) =0;

    inline void clear() { m_data_storage.clear(); }
    inline void shrink_to_fit() { m_data_storage.shrink_to_fit(); }

    inline const void * __restrict__ data_pointer() const { return onika::cuda::vector_data(m_data_storage); }
    inline void * __restrict__ data_pointer() { return onika::cuda::vector_data(m_data_storage); }
    
    template<class T> inline T * __restrict__ data() { return reinterpret_cast<T *>(data_pointer()); }
    template<class T> inline const T * __restrict__ data() const { return reinterpret_cast<const T *>(data_pointer()); }
    
  protected:
    onika::memory::CudaMMVector<uint8_t> m_data_storage;
  };

  template<class fid>
  class FlatArrayAdapter : public FlatArrayDescriptor
  {
  public:
    using field_id_t = onika::soatl::FieldId<fid>;
    using value_t = typename field_id_t::value_type;
    using pointer_t = value_t * __restrict__ ;
    using const_pointer_t = const value_t * __restrict__ ;
    
    const char* name() const override final
    {
      return field_id_t::name();
    }
    
    const char* short_name() const override final
    {
      return field_id_t::short_name();
    }
    
    std::string type_name() const override final
    {
      return typeid(value_t).name();
    }
    
    size_t size() const override final
    {
      assert( m_data_storage.size() % sizeof(value_t) == 0 ); return m_data_storage.size() / sizeof(value_t);
    }
    
    void resize(size_t sz) override final
    {
      m_data_storage.resize( sz * sizeof(value_t) );
    }
    
    inline pointer_t data()
    {
      return this->FlatArrayDescriptor::data<value_t>();
    }
    
    inline const_pointer_t data() const
    {
      return this->FlatArrayDescriptor::data<value_t>();
    }
  };  

} // end of namespace exanb

