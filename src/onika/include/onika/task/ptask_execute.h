#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor_impl.h>
#include <onika/task/parallel_task_queue.h>
#include <onika/flat_tuple.h>
#include <onika/zcurve.h>
#include <utility>
#include <type_traits>

#include <onika/cuda/cuda.h>

namespace onika
{
  namespace task
  {

    // single task compiler
    template<class _Span, class _TaskFunc, class _CostFunc, bool _CudaEnabled, class... _Accessor>
    static inline size_t ptask_execute_kernel( const PTaskProxy<_Span,SINGLE_TASK,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc,_CudaEnabled>& proxy , typename _Span::coord_t = {})
    {
      static_assert( ! std::is_same_v<_TaskFunc,NullTaskFunctor> , "cannot compile task kernel unless a task lambda is provided" );
      static_assert( _Span::ndims==0 , "0-D Span required for single tasks" );
      assert( proxy.m_ptask != nullptr );
      ptask_apply_accessors( proxy.m_func, proxy.m_accs , std::make_index_sequence<sizeof...(_Accessor)>{} );
      return 1;
    }

    // single fulfill task compiler
    template<class _Span, class _TaskFunc, class _CostFunc, bool _CudaEnabled, class... _Accessor>
    static inline size_t ptask_execute_kernel( const PTaskProxy<_Span,SINGLE_FULFILL_TASK,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc,_CudaEnabled>& proxy , typename _Span::coord_t = {} )
    {
      static_assert( ! std::is_same_v<_TaskFunc,NullTaskFunctor> , "cannot compile task unless a task lambda is provided" );
      static_assert( _Span::ndims==0 , "0-D Span required for single tasks" );
      assert( proxy.m_ptask != nullptr );
      assert( proxy.m_ptask->m_fulfilled_ptask != nullptr );
      ptask_apply_accessors_fulfill( proxy.m_ptask->m_fulfilled_ptask, proxy.m_func, proxy.m_accs , std::make_index_sequence<sizeof...(_Accessor)>{} );
      return 1;
    }

    // parallel for task compiler
    template<class _Span, class _TaskFunc, class _CostFunc, bool _CudaEnabled, class... _Accessor >
    ONIKA_HOST_DEVICE_FUNC 
    static inline size_t ptask_execute_kernel( const PTaskProxy<_Span,PARALLEL_FOR,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc,_CudaEnabled>& proxy , typename _Span::coord_t c )
    {
      using coord_t = typename _Span::coord_t;
      static_assert( ! std::is_same_v<_TaskFunc,NullTaskFunctor> , "cannot compile task unless a task lambda is provided" );
      static_assert( _Span::grainsize >= 1 , "null or negative grainsize not allowed" );
      static_assert( lambda_is_compatible_with_v< decltype(proxy.m_func) , void , coord_t , decltype( std::declval<_Accessor>() .at(coord_t{}) ) ... >
                   , "Functor argument does not have requested member method" );

      if constexpr ( _Span::grainsize > 1 )
      {
        //static_assert( is_po2_v<_Span::grainsize> , "only support power of 2 grain size" );
        static constexpr size_t Nd = _Span::ndims;
        size_t executed_elements = 0;          
        coord_t x = ZeroArray<size_t,Nd>::zero;
        while( x[Nd-1] < _Span::grainsize )
        {
          auto gc = array_add( c , x );
          if( proxy.m_span.inside( gc ) )
          {
            ptask_apply_accessors( proxy.m_func, proxy.m_accs , gc , std::make_index_sequence<sizeof...(_Accessor)>{} );
            ++ executed_elements;
          }
          ++ x[0];
          size_t i = 0;
          while( i<(Nd-1) && x[i]==_Span::grainsize ) { x[i]=0; ++i; ++x[i]; }
        }
        return executed_elements;
      }
      
      if constexpr ( _Span::grainsize == 1 )
      {
        assert( proxy.m_span.inside( c ) );
        ptask_apply_accessors( proxy.m_func, proxy.m_accs , c , std::make_index_sequence<sizeof...(_Accessor)>{} );
        return 1;
      }
      
      ONIKA_CU_ABORT();
      return 0;
    }

    // parallel for task cost function compiler
    template<class _Span, TaskTypeEnum _TaskType, class _TaskFunc, class _CostFunc, bool _CudaEnabled, class... _Accessor >
    ONIKA_HOST_DEVICE_FUNC 
    static inline ParallelTaskCostInfo ptask_execute_costfunc( const PTaskProxy<_Span,_TaskType,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc,_CudaEnabled>& proxy , typename _Span::coord_t c )
    {
      using coord_t = typename _Span::coord_t;
      static_assert( lambda_is_compatible_with_v<_CostFunc,uint64_t,coord_t> , "cost function does not have exepected call arguments and return type" );
      static constexpr unsigned int Nd = _Span::ndims;
      //assert( proxy.m_ptask != nullptr );
      if constexpr ( Nd > 0 )
      {
        if constexpr ( _Span::grainsize > 1 )
        {
          uint64_t total_cost = 0;
          int count = 0;
          int skipped = 0;
          coord_t x = ZeroArray<size_t,Nd>::zero;
          while( x[Nd-1] < _Span::grainsize )
          {
            auto gc = array_add( c , x );
            if( proxy.m_span.inside( gc ) )
            {
              uint64_t cost = 1; // default value, used when cost function is the trivial one
              if constexpr ( ! is_null_cost_func_v< decltype(proxy.m_cost) > ) { cost = proxy.m_cost( gc ); }
              if( cost == 0 ) ++skipped;
              total_cost += cost;
              ++ count;
            }
            ++ x[0];
            size_t i = 0;
            while( i<(Nd-1) && x[i]==_Span::grainsize ) { x[i]=0; ++i; ++x[i]; }
          }
          return { total_cost , count , skipped };
        }        
        if constexpr ( _Span::grainsize == 1 )
        {
          assert( proxy.m_span.inside( c ) );
          uint64_t cost = 1; // default value, used when cost function is the trivial one
          if constexpr ( ! is_null_cost_func_v< decltype(proxy.m_cost) > ) { cost = proxy.m_cost(c); }
          return { cost , 1 , cost==0 };
        }
      }
      ONIKA_CU_ABORT(); return {};
    }

  }
  
}


