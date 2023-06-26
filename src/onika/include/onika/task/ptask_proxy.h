#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_queue.h>
#include <onika/flat_tuple.h>
#include <onika/dac/box_span.h>
#include <utility>
#include <type_traits>

#include <onika/task/ptask_ops_proto.h>
#include <onika/task/ptask_proxy_proto.h>
#include <onika/task/tag_utils.h>
#include <onika/oarray.h>

namespace onika
{
  namespace task
  {
    
    template<class T> struct IsPTaskProxy : public std::false_type {};
    template<class _Span, TaskTypeEnum _TaskType, class _AccessorTuple , class _TaskFunc , class _CostFunc, bool _CudaEnabled >
    struct IsPTaskProxy< PTaskProxy<_Span,_TaskType,_AccessorTuple,_TaskFunc,_CostFunc,_CudaEnabled> > : public std::true_type {};

    using ObliviousPTaskProxy = PTaskProxy<dac::box_span_t<0>,TASK_TYPE_OBLIVIOUS,FlatTuple<>,NullTaskFunctor,NullTaskCostFunc<0>,false>;
    
    template<class _Span, TaskTypeEnum _TaskType , class _TaskFunc , class _CostFunc, bool _CudaEnabled , class... _Accessor>
    struct PTaskProxy< _Span, _TaskType , FlatTuple<_Accessor...> , _TaskFunc , _CostFunc , _CudaEnabled >
    {
      static inline constexpr TaskTypeEnum TaskType = _TaskType;
      static inline constexpr bool CudaEnabled = _CudaEnabled;
      using Span = _Span;
      //using Allocator = typename TaskPoolItem::TaskAllocator;
      using AccessorTuple = FlatTuple<_Accessor...>;
      using TaskFunc = _TaskFunc;
      using CostFunc = _CostFunc;
      
      ParallelTaskQueue* m_queue = nullptr;
      ParallelTask* m_ptask = nullptr;
      const char* m_tag = nullptr;
      uint64_t m_flags = 0;
      Span m_span;
      AccessorTuple m_accs;
      TaskFunc m_func;
      CostFunc m_cost;

      static_assert( lambda_is_compatible_with_v<CostFunc,uint64_t,typename _Span::coord_t> , "cost function does not have exepected call arguments and return type" );

      PTaskProxy() = default;
      PTaskProxy(const PTaskProxy&) = default;
      inline PTaskProxy(PTaskProxy && ptp)
        : m_queue( ptp.m_queue )
        , m_ptask( ptp.m_ptask )
        , m_tag( ptp.m_tag )
        , m_flags( ptp.m_flags )
        , m_span( std::move( ptp.m_span ) )
        , m_accs( std::move( ptp.m_accs ) )
        , m_func( std::move( ptp.m_func ) )
        , m_cost( std::move( ptp.m_cost ) )
      {
        ptp.reset();
      }

      inline PTaskProxy( ParallelTaskQueue* ptq, ParallelTask* pt, const char* tag, uint64_t flags, const Span & sp = Span{}, AccessorTuple && at = AccessorTuple{} , TaskFunc && f = TaskFunc{} , CostFunc && cf = CostFunc{} )
        : m_queue(ptq)
        , m_ptask(pt)
        , m_tag(tag)
        , m_flags(flags)
        , m_span(sp)
        , m_accs(std::move(at))
        , m_func(std::move(f))
        , m_cost(std::move(cf))
      {}

      template<class _Func>
      inline PTaskProxy<_Span,TaskType,AccessorTuple,_Func,CostFunc,CudaEnabled> operator * ( _Func && f )
      {
        static_assert( std::is_same_v<TaskFunc,NullTaskFunctor> , "A task functor has already been attached to parallel task" );
        return PTaskProxy<_Span,TaskType,AccessorTuple,_Func,CostFunc,CudaEnabled>( 
                  m_queue
                , m_ptask
                , m_tag
                , m_flags
                , std::move(m_span) 
                , std::move(m_accs) 
                , std::move(f) 
                , std::move(m_cost) );
      }

      template<bool CE>
      inline PTaskProxy<_Span,TaskType,AccessorTuple,TaskFunc,CostFunc,CE> operator / ( proxy_set_cuda_t<CE> )
      {
        return PTaskProxy<_Span,TaskType,AccessorTuple,TaskFunc,CostFunc,CE>( 
                  m_queue
                , m_ptask
                , m_tag
                , m_flags
                , std::move(m_span) 
                , std::move(m_accs) 
                , std::move(m_func) 
                , std::move(m_cost) );        
      }

      template<class _Func>
      inline PTaskProxy<_Span,TaskType,AccessorTuple,TaskFunc,_Func,CudaEnabled> operator / ( _Func && f )
      {
        static_assert( std::is_same_v<CostFunc,NullTaskCostFunc<_Span::ndims> > , "A cost functor has already been attached to parallel task" );
        static_assert( lambda_is_compatible_with_v<decltype(f),uint64_t,typename _Span::coord_t> , "cost function does not have exepected signature" );
        return PTaskProxy<_Span,TaskType,AccessorTuple,TaskFunc,_Func,CudaEnabled>( 
                  m_queue
                , m_ptask
                , m_tag
                , m_flags
                , std::move(m_span) 
                , std::move(m_accs) 
                , std::move(m_func) 
                , std::move(f) );
      }

      inline ParallelTask* first_ptask() const { return m_ptask; }
      inline ParallelTask* last_ptask() const { return m_ptask; }
      inline ParallelTaskQueue* ptask_queue() const { return m_queue; }
      inline void attach_to_queue(ParallelTaskQueue* ptq)
      {
        // _Pragma("omp critical(dbg_mesg)") std::cout<<"ptask "<<tag_filter_out_path(m_tag)<<" attach to queue"<<std::endl;
        m_queue = ptq;
      }

      inline void allocate_ptask()
      {
        assert( m_queue != nullptr );
        assert( m_ptask == nullptr );
        // _Pragma("omp critical(dbg_mesg)") std::cout<<"allocate ptask "<<tag_filter_out_path(m_tag)<<std::endl;
        auto && [buf,nr,ny,sw] = m_queue->allocator().allocate_nofail( sizeof(ParallelTask) );
        m_ptask = new(buf) ParallelTask(m_flags /*, m_tag*/ );
      }

      inline void finalize()
      {
        if constexpr ( ! std::is_same_v<_TaskFunc,NullTaskFunctor> )
        {
          if( m_queue != nullptr )
          {
            if( m_ptask == nullptr ) { allocate_ptask(); }
            assert( m_ptask != nullptr );
            if( m_ptask->m_ptexecutor == nullptr ) { ptask_compile( std::move(*this) ); }        
          }
        }
      }

      inline void reset() { m_queue=nullptr; m_ptask=nullptr; m_tag=nullptr; m_flags=0; }

      inline void execute()
      {
        if constexpr ( ! std::is_same_v<_TaskFunc,NullTaskFunctor> )
        {
          if( m_queue != nullptr )
          {
            // _Pragma("omp critical(dbg_mesg)") std::cout<<"PTaskProxy::execute() => enqueue ptask "<<tag_filter_out_path(m_tag)<<std::endl;
            m_queue->enqueue_ptask( m_ptask );
          }
        }
      }

      inline ~PTaskProxy() { finalize(); execute(); reset(); }
      template<class S> S& print(S& out,int d=0)
      {
        out<<tag_filter_out_path(m_tag);
        if(d==0) out<<"\n";
        return out;
      }

      template<class PTask2>
      inline auto operator >> ( PTask2 && t2)
      {
        return ptask_sequence_after( std::move(*this) , std::move(t2) );
      }

      template<class PTask2>
      inline auto operator || ( PTask2 && t2 )
      {
        //static_assert( IsPTaskProxy<PTask2>::value , "only PTaskProxy is allowed here" );
        return ptask_schedule_after( std::move(*this) , std::move(t2) );
      }
      
    };

  }
  
}


