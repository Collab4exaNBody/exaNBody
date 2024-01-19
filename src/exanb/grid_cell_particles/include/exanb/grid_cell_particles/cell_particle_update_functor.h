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

#include <onika/soatl/field_tuple.h>
#include <onika/soatl/field_id.h>
#include <onika/cuda/cuda.h>
#include <onika/integral_constant.h>

namespace exanb
{

  struct UpdateValueAdd
  {
    template< class T , bool ThreadSafe >
    ONIKA_HOST_DEVICE_FUNC
    static inline void update_add_value( T& x , const T& a , onika::BoolConst<ThreadSafe> ts )
    {
      if constexpr ( ThreadSafe )
      {
        ONIKA_CU_ATOMIC_ADD( x , a , ONIKA_CU_MEM_ORDER_RELAXED );
      }
      if constexpr ( ! ThreadSafe )
      {
        x += a;
      }
    }

    template< bool ThreadSafe >
    ONIKA_HOST_DEVICE_FUNC
    static inline void update_add_value( Vec3d& upd , const Vec3d& in , onika::BoolConst<ThreadSafe> ts )
    {
      update_add_value( upd.x , in.x , ts );
      update_add_value( upd.y , in.y , ts );
      update_add_value( upd.z , in.z , ts );
    }

    template< bool ThreadSafe >
    ONIKA_HOST_DEVICE_FUNC
    static inline void update_add_value( Mat3d& upd , const Mat3d& in , onika::BoolConst<ThreadSafe> ts )
    {
      update_add_value( upd.m11 , in.m11 , ts );
      update_add_value( upd.m12 , in.m12 , ts );
      update_add_value( upd.m13 , in.m13 , ts );
      update_add_value( upd.m21 , in.m21 , ts );
      update_add_value( upd.m22 , in.m22 , ts );
      update_add_value( upd.m23 , in.m23 , ts );
      update_add_value( upd.m31 , in.m31 , ts );
      update_add_value( upd.m32 , in.m32 , ts );
      update_add_value( upd.m33 , in.m33 , ts );
    }

    template<class CellsT, bool ThreadSafe, typename... field_ids>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator()( CellsT cells, size_t cell_i, size_t p_i, const onika::soatl::FieldTuple<field_ids...>& in , onika::BoolConst<ThreadSafe> ts ) const
    {
      ( ... , ( update_add_value( cells[cell_i][ onika::soatl::FieldId<field_ids>() ][p_i] , in[ onika::soatl::FieldId<field_ids>() ] , ts ) ) );
    }

    template<bool ThreadSafe=false>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator() (double& upd, const double& in, onika::BoolConst<ThreadSafe> ts = {} ) const
    {
      update_add_value( upd , in , ts );
    }
  };

  struct UpdateValueAssertEqual
  {
    template<class T1, class T2>
    ONIKA_HOST_DEVICE_FUNC
    static inline void assert_field_equal(const T1& t1, const T2& t2)
    {
      assert( t1 == t2 );
    }
  
    template<class CellsT, bool ThreadSafe, typename... field_ids>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator()( CellsT cells, size_t cell_i, size_t p_i, const onika::soatl::FieldTuple<field_ids...>& in , onika::BoolConst<ThreadSafe> ) const
    {
      ( ... , ( assert_field_equal( cells[cell_i][onika::soatl::FieldId<field_ids>{}][p_i] , in[onika::soatl::FieldId<field_ids>{}] ) ) );
    }

    template<bool ThreadSafe=false>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator() ( const double& upd, const double& in, onika::BoolConst<ThreadSafe> = {}) const
    {
      assert( upd == in );
    }
  };

}

