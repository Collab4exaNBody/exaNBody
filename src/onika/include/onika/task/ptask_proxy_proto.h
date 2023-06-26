#pragma once

#include <onika/task/ptask_ops_proto.h>
#include <onika/task/ptask_null_functor.h>
#include <onika/task/parallel_task_cost.h>

namespace onika
{
  namespace task
  {

    enum TaskTypeEnum
    {
      SINGLE_TASK,
      SINGLE_FULFILL_TASK,
      PARALLEL_FOR,
      CUDA_PARALLEL_FOR,
      TASK_TYPE_OBLIVIOUS
    };

    template< class _Span
            , TaskTypeEnum _TaskType
            , class _AccessorTuple 
            , class _TaskFunc = NullTaskFunctor 
            , class _CostFunc = NullTaskCostFunc<_Span::ndims> 
            , bool _CudaEnabled = false
            > struct PTaskProxy;
  }
  
}


