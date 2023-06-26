#pragma once

namespace onika
{
  namespace task
  {
    
    enum PTaskBinOp
    {
      SEQUENCE_AFTER ,
      SCHEDULE_AFTER
    };

    template<PTaskBinOp _Op, class PTask1, class PTask2> struct PTaskBinExpr;
    template<class PTask1, class PTask2> static inline PTaskBinExpr<SEQUENCE_AFTER,PTask1,PTask2> ptask_sequence_after( PTask1 && t1 , PTask2 && t2 );
    template<class PTask1, class PTask2> static inline PTaskBinExpr<SCHEDULE_AFTER,PTask1,PTask2> ptask_schedule_after( PTask1 && t1 , PTask2 && t2 );
    
    template<bool C> struct proxy_set_cuda_t {};
    static inline constexpr proxy_set_cuda_t<true> proxy_cuda_enable = {};
    static inline constexpr proxy_set_cuda_t<false> proxy_cuda_disable = {};
  }
}

