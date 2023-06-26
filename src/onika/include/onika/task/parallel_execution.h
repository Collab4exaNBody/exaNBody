#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_queue.h>
#include <onika/task/ptask_functional.h>
#include <onika/task/ptask_proxy.h>
#include <onika/task/ptask_compile.h>
#include <onika/task/ptask_ops.h>

#include <onika/flat_tuple.h>
#include <onika/macro_utils.h>
#include <onika/integral_constant.h>

#include <onika/cuda/cuda.h>

namespace onika
{
  namespace task
  {

    template<class _Span, bool _CudaEnabled, class... _Accessor >
    static inline auto
    parallel_for(uint64_t flags, const _Span & sp, const char* tag, BoolConst<_CudaEnabled>, const _Accessor& ... accessors)
    {
      constexpr auto conversion_defaults_to_read_only_if_not_0d = std::integral_constant< bool , (_Span::ndims>=1) >{};
      auto at = dac::auto_dac_tuple_convert( conversion_defaults_to_read_only_if_not_0d , accessors... ); // parameters that are not accessors are converted to read-only scalar accessors
      return PTaskProxy<_Span,PARALLEL_FOR,decltype(at),NullTaskFunctor,NullTaskCostFunc<_Span::ndims>,_CudaEnabled> ( nullptr , nullptr , tag , flags , sp , std::move(at) );
    }

    template<class... _Accessor >
    static inline auto
    single_task(uint64_t flags, const char* tag, _Accessor& ... accessors)
    {
      using _Span = dac::box_span_t<0>;
      constexpr auto conversion_defaults_to_rw = std::false_type{};
      auto at = dac::auto_dac_tuple_convert( conversion_defaults_to_rw , accessors... );
      return PTaskProxy<_Span,SINGLE_TASK,decltype(at)> ( nullptr , nullptr , tag , flags , _Span{} , std::move(at) );
    }

    template<class... _Accessor >
    static inline auto
    fulfill_task(uint64_t flags, const char* tag, _Accessor& ... accessors)
    {
      using _Span = dac::box_span_t<0>;
      constexpr auto conversion_defaults_to_rw = std::false_type{};
      auto at = dac::auto_dac_tuple_convert( conversion_defaults_to_rw , accessors... );
      return PTaskProxy<_Span,SINGLE_FULFILL_TASK,decltype(at)> ( nullptr , nullptr , tag , flags , _Span{} , std::move(at) );
    }

  }
  
}

// auto convert to raw parameter to accessor's returned type
#define _ONIKA_ACC_TO_REF(a) ::onika::dac::auto_conv_dac_subset_t<(decltype(span)::ndims>1),decltype(a)> a
#define _ONIKA_ACC_TO_REF_RW(a) ::onika::dac::auto_conv_dac_subset_t<false,decltype(a)> a

// accessor or raw data to accessor reference type
#define _ONIKA_MK_ACC_REF_CTX_FUNC1(a) ::onika::dac::auto_dac_convert_t<(decltype(span)::ndims>1),decltype(a)> _arg_##a
#define _ONIKA_MK_ACC_REF_CTX_FUNC2(a) decltype(_arg_##a) a;
#define _ONIKA_MK_ACC_REF_CTX_FUNC3(a) _arg_##a
#define _ONIKA_MK_ACC_REF_CTX_FUNC4(a) ::onika::dac::auto_dac_convert(std::integral_constant<bool,(decltype(span)::ndims>1)>{},a)
#define _ONIKA_MK_ACC_REF_CTX(...) \
[]( EXPAND_WITH_FUNC(_ONIKA_MK_ACC_REF_CTX_FUNC1 OPT_COMMA_VA_ARGS(__VA_ARGS__)) ) \
{ \
  struct { EXPAND_WITH_FUNC_NOSEP(_ONIKA_MK_ACC_REF_CTX_FUNC2 OPT_COMMA_VA_ARGS(__VA_ARGS__)) } _acc_ctx { EXPAND_WITH_FUNC(_ONIKA_MK_ACC_REF_CTX_FUNC3 OPT_COMMA_VA_ARGS(__VA_ARGS__)) };\
  return _acc_ctx; \
} \
( EXPAND_WITH_FUNC(_ONIKA_MK_ACC_REF_CTX_FUNC4 OPT_COMMA_VA_ARGS(__VA_ARGS__)) )

#define onika_parallel_for(span,...) \
::onika::task::parallel_for( 0 , \
              span , \
              "parallel_for@" __FILE__ ":" ONIKA_STR(__LINE__) , \
              FalseType{} \
              OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
* [=] ONIKA_HOST_DEVICE_FUNC ( decltype(std::declval<decltype(span)>().lower_bound) item_coord EXPAND_WITH_FUNC_PREPEND_COMMA(_ONIKA_ACC_TO_REF OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ) -> void

#define onika_detached_parallel_for(span,...) \
::onika::task::parallel_for( ::onika::task::PTASK_FLAG_DETACHED , \
              span , \
              "detached_parallel_for@" __FILE__ ":" ONIKA_STR(__LINE__) , \
              FalseType{} \
              OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
* [=]( decltype(std::declval<decltype(span)>().lower_bound) item_coord EXPAND_WITH_FUNC_PREPEND_COMMA(_ONIKA_ACC_TO_REF OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ) -> void

#define onika_task(...) \
::onika::task::single_task( 0 , \
            "task@" __FILE__ ":" ONIKA_STR(__LINE__) \
            OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
* [=]( EXPAND_WITH_FUNC(_ONIKA_ACC_TO_REF_RW OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ) -> void

#define onika_fulfill_task(...) \
::onika::task::fulfill_task( ::onika::task::PTASK_FLAG_FULFILL,"fulfill_task@" __FILE__ ":" ONIKA_STR(__LINE__) \
              OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
* [=]( ::onika::task::ParallelTaskExecutionContext&& onika_ctx EXPAND_WITH_FUNC_PREPEND_COMMA(_ONIKA_ACC_TO_REF_RW OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ) -> void

