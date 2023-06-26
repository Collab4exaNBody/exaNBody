#pragma once

#include <onika/task/parallel_execution.h>
#include <onika/dac/box_span.h>
#include <onika/flat_tuple.h>
#include <onika/macro_utils.h>
#include <onika/cuda/cuda.h>

namespace onika
{
  namespace task
  {
  
    template <class F, class AT, size_t... Is>
    __global__ void cuda_ptask_kernel_1d( const F& f , size_t first_i, size_t gs, size_t n , const AT& at , std::index_sequence<Is...> )
    {
      size_t i = blockDim.x * blockIdx.x + threadIdx.x;
      size_t j = first_i + i;
      if ( i < gs && j < n )
      {
        typename flat_tuple_element_t<AT,0>::item_coord_t c = {j};
        f( c , at.get(tuple_index<Is>).at(c) ... );
      }
    }

    // cuda parallel for task compiler
    // TODO: set m_task_space=PTASK_EXECUTE_CUDA and m_task_signature=PTASK_MONOLITHIC
    template<class _Span, class _TaskFunc, class _CostFunc, class... _Accessor >
    static inline auto ptask_compile_kernel( const PTaskProxy<_Span,CUDA_PARALLEL_FOR,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc>& proxy )
    {
      using AccessorTuple = FlatTuple<_Accessor...>;
      using coord_t = typename _Span::coord_t;
      static_assert( std::is_same_v<decltype(proxy.m_func(coord_t{},std::declval<decltype( std::declval<_Accessor>() .at(coord_t{}))>()...)),void> , "Functor argument does not have requested member method" );
      static_assert( _Span::ndims == 1 , "only works for 1D kernels by now" );
      return [ f=proxy.m_func , at=proxy.m_accs , st=proxy.m_queue->cuda_streams() ]( ParallelTask* pt ) -> void
      {
        assert( ! pt->m_detached );
        assert( ! pt->m_gen_omp_out_dep );
        /*if( pt->m_task_space == PTASK_EXECUTE_HOST )
        {
#         pragma omp taskgroup
          {
            for(size_t i=0;i<pt->span().box_size[0];i++)
            {
              typename flat_tuple_element_t<AccessorTuple,0>::item_coord_t c = {i+pt->span().lower_bound[0]};
#             pragma omp task default(none) firstprivate(c) shared(f,at)
              ptask_apply_accessors(f,at,c,std::make_index_sequence<sizeof...(_Accessor)>{});
            }
          }
        }
        else*/
        {
          int threadsPerBlock = 64; // 256;
          int n = pt->span().lower_bound[0] + pt->span().box_size[0];
          int blocksPerGrid =( n + threadsPerBlock - 1) / threadsPerBlock;
          cuda_ptask_kernel_1d<<< blocksPerGrid, threadsPerBlock , 0, st[0] >>> (f,pt->span().lower_bound[0],pt->span().box_size[0],n, at , std::make_index_sequence<sizeof...(_Accessor)>{} );
          cudaStreamSynchronize ( st[0] );
          int n_executed = pt->span().box_size[0];
          pt->account_completed_task( n_executed );
        }
      };
    }

    template<class _Span, class... _Accessor >
    static inline auto
    cuda_parallel_for(uint64_t flags, const _Span& sp, const char* tag, const _Accessor& ... accessors)
    {
      constexpr auto convesion_defaults_to_read_only_if_not_0d = std::integral_constant< bool , (_Span::ndims>=1) >{};
      auto at = dac::auto_dac_tuple_convert( convesion_defaults_to_read_only_if_not_0d , accessors... ); // parameters that are not accessors are converted to read-only scalar accessors
      return PTaskProxy<_Span,CUDA_PARALLEL_FOR,decltype(at)> ( nullptr , nullptr , tag , flags , sp , std::move(at) );
    }

  }
  
}


#ifdef __CUDA_ARCH__
#define onika_simd_for(i,s,e,l) \
      for( ssize_t j=0 , i=s+threadIdx.x ; j<e ; j+=blockDim.x, i+=blockDim.x ) if(i<n)
#endif

#define onika_cuda_parallel_for(span,...) \
::onika::task::cuda_parallel_for(0,span,"cuda_parallel_for@" __FILE__ ":" ONIKA_STR(__LINE__) OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
* [=] ONIKA_HOST_DEVICE_FUNC ( decltype(std::declval<decltype(span)>().lower_bound) item_coord EXPAND_WITH_FUNC_PREPEND_COMMA(_ONIKA_ACC_TO_REF OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ) -> void

