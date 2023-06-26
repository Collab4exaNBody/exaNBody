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

    template<class _Span, TaskTypeEnum _TaskType, class _TaskFunc, class _CostFunc, bool _CudaEnabled, class... _Accessor>
    static inline void ptask_compile( PTaskProxy<_Span,_TaskType,FlatTuple<_Accessor...>,_TaskFunc,_CostFunc,_CudaEnabled> && proxy )
    {
      assert( proxy.m_queue != nullptr );
      if( proxy.m_ptask == nullptr ) { proxy.allocate_ptask(); }
      assert( proxy.m_ptask->m_ptexecutor == nullptr );
      auto * pt = proxy.m_ptask;
      auto * ptq = proxy.m_queue;
      pt->m_ptexecutor = ParallelTaskExecutor::ptexecutor_adapter( std::move(proxy) );
      // we let functors be "swallowed", but restore ptask and queue to correctly enqueue and connect tasks
      proxy.m_ptask = pt;
      proxy.m_queue = ptq;
      //std::cout<<"associate ptexecutor@"<<(void*)proxy.m_ptask->m_ptexecutor.get()<<" to ptask@"<<(void*)proxy.m_ptask<<"\n";
    }

  }
  
}


