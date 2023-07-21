#pragma once

#include <onika/soatl/field_tuple.h>
#include <onika/soatl/field_id.h>
#include <onika/cuda/cuda.h>
#include <onika/integral_constant.h>

namespace exanb
{

  struct UpdateValueAdd
  {
    template<class CellsT, bool ThreadSafe, typename... field_ids>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator()( CellsT cells, size_t cell_i, size_t p_i, const onika::soatl::FieldTuple<field_ids...>& in , onika::BoolConst<ThreadSafe> ) const
    {
      if constexpr ( ThreadSafe )
      {
        ( ... , ( ONIKA_CU_ATOMIC_ADD( cells[cell_i][ onika::soatl::FieldId<field_ids>() ][p_i] , in[ onika::soatl::FieldId<field_ids>() ] , ONIKA_CU_MEM_ORDER_RELAXED ) ) );
      }
      else
      {
        ( ... , ( cells[cell_i][ onika::soatl::FieldId<field_ids>() ][p_i] += in[ onika::soatl::FieldId<field_ids>() ] ) );
      }
    }

    template<bool ThreadSafe=false>
    ONIKA_HOST_DEVICE_FUNC
    inline void operator() (double& upd, const double& in, onika::BoolConst<ThreadSafe> = {} ) const
    {
      if constexpr ( ThreadSafe )
      {
        ONIKA_CU_ATOMIC_ADD( upd , in , ONIKA_CU_MEM_ORDER_RELAXED );
      }
      else
      {
        upd += in;
      }
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

