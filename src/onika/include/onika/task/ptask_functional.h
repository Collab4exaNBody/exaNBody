#pragma once

#include <onika/task/parallel_task.h>
#include <onika/flat_tuple.h>
#include <onika/dac/dac.h>
#include <utility>
#include <onika/task/ptask_null_functor.h>
#include <onika/cuda/cuda.h>

namespace onika
{
  namespace task
  {
    template <class F, class AT, class C, size_t... I>
    ONIKA_HOST_DEVICE_FUNC static inline void ptask_apply_accessors( const F& f, const AT & args , const C& c, std::index_sequence<I...>)
    {
      static_assert( is_onika_array_v<C> , "item coordinate must be a onika::oarray<> instance" );
      f( c , args.get(tuple_index<I>).at(c) ... );
    }

    template <class F, class AT, size_t... I>
    static inline void ptask_apply_accessors( const F& f, const AT & args, std::index_sequence<I...>)
    {
      static_assert( ( ... && ( flat_tuple_element_t<AT,I>::ND == 0) ) , "non 0D accessors not allowed whithout coordinate" );
      static_assert( ( ... && ( dac::is_data_access_controler_v< decltype(args.get(tuple_index<I>)) > ) ) , "tuple contain a non-DataAccessControler type" );
      static_assert( ( ... && ( ! dac::is_data_access_controler_v< decltype(args.get(tuple_index<I>).at()) > ) ) , "value accessor must not be a DataAccessControler type" );
      f( args.get(tuple_index<I>).at() ... );
    }

    template <class F, class AT, /*class FFAccs,*/ size_t... I>
    static inline void ptask_apply_accessors_fulfill( ParallelTask* pt, const F& f, const AT & args, /*const FFAccs& ffaccs,*/ std::index_sequence<I...>)
    {
      static_assert( ( ... && ( flat_tuple_element_t<AT,I>::ND == 0) ) , "non 0D accessors not allowed whithout coordinate" );
//      auto onika_fulfill = [pt]( auto c = oarray_t<size_t,0>{} ) { pt->completion_notify(c); };
//      auto onika_fulfill_span = [pt]() -> const dac::abstract_box_span_t& { return pt->m_span; };
      f( ParallelTaskExecutionContext{pt/*,ffaccs*/} , args.get(tuple_index<I>).at() ... );
    }

  }
  
}


