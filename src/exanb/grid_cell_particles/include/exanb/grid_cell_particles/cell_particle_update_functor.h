#pragma once

#include <onika/soatl/field_tuple.h>
#include <onika/soatl/field_id.h>

namespace exanb
{

  struct UpdateValueAdd
  {
    template<class FullTupleT, typename... field_ids>
    inline void operator()( FullTupleT & upd, const onika::soatl::FieldTuple<field_ids...>& in) const
    {
      ( ... , ( upd[ onika::soatl::FieldId<field_ids>() ] += in[ onika::soatl::FieldId<field_ids>() ] ) );
    }

    inline void operator() (double& upd, const double& in) const
    {
      upd += in;
    }
  };

  struct UpdateValueAssertEqual
  {
    template<class T1, class T2, class F>
    static inline void assert_field_equal(const T1& t1, const T2& t2, F f)
    {
      assert( t1[f] == t2[f] );
    }
  
    template<class FullTupleT, typename... field_ids>
    inline void operator()( const FullTupleT & upd, const onika::soatl::FieldTuple<field_ids...>& in) const
    {
      ( ... , ( assert_field_equal(upd,in,onika::soatl::FieldId<field_ids>{}) ) );
    }

    inline void operator() ( const double& upd, const double& in) const
    {
      assert( upd == in );
    }
  };

}

