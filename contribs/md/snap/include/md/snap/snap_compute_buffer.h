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

#include <onika/cuda/cuda.h>

namespace md
{

  template<class SizeT , class RealT = double >
  struct SnapXSTemporaryComplexArray
  {
    using real_t = RealT;
    static inline constexpr int Size = 0;
  
    SizeT m_array_size = {};
    real_t * __restrict__ m_ptr = nullptr;
    
    ONIKA_HOST_DEVICE_FUNC
    inline void reset()
    {
      m_array_size = SizeT{};
      m_ptr = nullptr;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void init( SizeT sz )
    {
      m_array_size = sz;
    }
    
    ONIKA_HOST_DEVICE_FUNC
    inline void alloc_array()
    {
      if( m_ptr == nullptr ) m_ptr = new real_t [ m_array_size * 2 ];
    }
    
    ONIKA_HOST_DEVICE_FUNC
    inline real_t * __restrict__ r()
    {
      alloc_array();
      return m_ptr;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline real_t * __restrict__ i()
    {
      alloc_array();
      return m_ptr + m_array_size;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline const real_t * __restrict__ r() const
    {
      assert( m_ptr != nullptr );
      return m_ptr;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline const real_t * __restrict__ i() const
    {
      assert( m_ptr != nullptr );
      return m_ptr + m_array_size;
    }

    
    ONIKA_HOST_DEVICE_FUNC
    inline ~SnapXSTemporaryComplexArray()
    {
      if( m_ptr != nullptr ) delete [] m_ptr;
    }
  };

  template<int _SZ, class RealT>
  struct SnapXSTemporaryComplexArray< onika::IntConst<_SZ> , RealT >
  {
    using real_t = RealT;
    
    static inline constexpr int Size = _SZ;
    real_t m_storage[ Size * 2 ];

    ONIKA_HOST_DEVICE_FUNC
    inline void reset()
    {
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void init( int sz )
    {
      assert( Size == sz );
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void alloc_array() {}

    ONIKA_HOST_DEVICE_FUNC
    inline real_t * __restrict__ r()
    {
      return m_storage;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline real_t * __restrict__ i()
    {
      return m_storage + Size;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline const real_t * __restrict__ r() const
    {
      return m_storage;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline const real_t * __restrict__ i() const
    {
      return m_storage + Size;
    }

  };

  template<class SnapConfT , class RealT = double>
  struct SnapXSForceExtStorage
  {  
    SnapXSTemporaryComplexArray< std::remove_cv_t< std::remove_reference_t< decltype( SnapConfT{}.idxu_max * SnapConfT{}.nelements ) > > , RealT > m_UTot_array = {};
    SnapXSTemporaryComplexArray< std::remove_cv_t< std::remove_reference_t< decltype( SnapConfT{}.idxu_max_alt * SnapConfT{}.nelements ) > > , RealT > m_Y_array    = {};
    // dynamically allocated only if needed for non specific cases (specialized implementations do not use these arrays)
    SnapXSTemporaryComplexArray<int,RealT> m_U_array    = {};
    SnapXSTemporaryComplexArray<int,RealT> m_DU_array   = {};

    ONIKA_HOST_DEVICE_FUNC
    inline void reset()
    {
      m_U_array.reset();
      m_UTot_array.reset();
      m_DU_array.reset();
      m_Y_array.reset();
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void init( const SnapConfT& snaconf )
    {
      m_U_array.init( snaconf.idxu_max );
      m_UTot_array.init( snaconf.idxu_max * snaconf.nelements );
      m_DU_array.init( snaconf.idxu_max * onika::IntConst<3>{} );
      m_Y_array.init( snaconf.idxu_max_alt * snaconf.nelements );
    }
  };

  template<class SnapConfT>
  struct SnapBSExtStorage
  {  
    SnapXSTemporaryComplexArray< std::remove_cv_t< std::remove_reference_t< decltype( SnapConfT{}.idxu_max                         ) > > > m_U_array    = {};
    SnapXSTemporaryComplexArray< std::remove_cv_t< std::remove_reference_t< decltype( SnapConfT{}.idxu_max * SnapConfT{}.nelements ) > > > m_UTot_array = {};
    SnapXSTemporaryComplexArray< std::remove_cv_t< std::remove_reference_t< decltype( SnapConfT{}.idxz_max * SnapConfT{}.ndoubles  ) > > > m_Z_array    = {};

    ONIKA_HOST_DEVICE_FUNC
    inline void reset()
    {
      m_U_array.reset();
      m_UTot_array.reset();
      m_Z_array.reset();
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void init( const SnapConfT& snaconf )
    {
      m_U_array.init( snaconf.idxu_max );
      m_UTot_array.init( snaconf.idxu_max * snaconf.nelements );
      m_Z_array.init( snaconf.idxz_max * snaconf.ndoubles );
    }
  };

}
