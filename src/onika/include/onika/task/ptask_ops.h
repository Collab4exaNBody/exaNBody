#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_queue.h>
#include <onika/task/ptask_ops_proto.h>

namespace onika
{
  namespace task
  {
    
    struct PTaskProxyQueueFlush
    {
      ParallelTaskQueue* ptq = nullptr;
      
      PTaskProxyQueueFlush() = default;
      PTaskProxyQueueFlush(const PTaskProxyQueueFlush&) = default;
      PTaskProxyQueueFlush(ParallelTaskQueue* q) : ptq(q) {}
      inline PTaskProxyQueueFlush(PTaskProxyQueueFlush && ptpqf) : ptq(ptpqf.ptq) { ptpqf.reset(); }
      inline ParallelTaskQueue* ptask_queue() const { return ptq; }
      inline void attach_to_queue(ParallelTaskQueue* q) { ptq = q; }
      static inline constexpr ParallelTask* first_ptask() { return nullptr; }
      static inline constexpr ParallelTask* last_ptask() { return nullptr; }
      static inline constexpr void finalize() {}
      inline void reset() { ptq = nullptr; }
      inline void execute()
      {
        if(ptq!=nullptr)
        {
          // _Pragma("omp critical(dbg_mesg)") std::cout<<"flush queue"<<std::endl;
          ptq->flush();
        }
      }
      ~PTaskProxyQueueFlush() { finalize(); execute(); reset(); }
      template<class S> S& print(S& out,int d=0)
      {
        out<<"flush";
        if(d==0) out<<"\n";
        return out;
      }
    };

    inline PTaskProxyQueueFlush flush() { return {}; }

    template<PTaskBinOp _Op, class PTask1, class PTask2 > struct PTaskBinExprConnector;
    template<class PTask1, class PTask2 >
    struct PTaskBinExprConnector<SEQUENCE_AFTER,PTask1,PTask2>
    {
      static inline void connect( PTask1& pt1 , PTask2& pt2 )
      {
        auto * lt1 = pt1.last_ptask();
        auto * ft2 = pt2.first_ptask();
        if( lt1 != nullptr && ft2 != nullptr )
        {
          while( lt1->m_sequence_after != nullptr ) { lt1 = lt1->m_sequence_after; }
          ft2->sequence_after( lt1 );        
        }
      }
    };
    template<class PTask1, class PTask2 >
    struct PTaskBinExprConnector<SCHEDULE_AFTER,PTask1,PTask2>
    {
      static inline void connect( PTask1& pt1 , PTask2& pt2 )
      {
        auto * lt1 = pt1.last_ptask();
        auto * ft2 = pt2.first_ptask();
        if( lt1 != nullptr && ft2 != nullptr )
        {
          ft2->schedule_after( lt1 );
          if( lt1->detached_task() && ft2->fulfills_task() )
          {
            ft2->fulfills( lt1 );
          }
        }
      }
    };

    template<PTaskBinOp _Op, class PTask1, class PTask2>
    struct PTaskBinExpr
    {
      static inline constexpr PTaskBinOp Op = _Op;
      PTask1 pt1;
      PTask2 pt2;

      inline PTaskBinExpr() = delete ;
      inline PTaskBinExpr(const PTaskBinExpr& ) = delete ;
      inline PTaskBinExpr(PTaskBinExpr && other ) : pt1(std::move(other.pt1)) , pt2(std::move(other.pt2)) {}
      inline PTaskBinExpr(PTask1 && opt1 , PTask2 && opt2 ) : pt1(std::move(opt1)) , pt2(std::move(opt2)) {}

      inline ParallelTask* first_ptask() const
      {
        auto * pt = pt1.first_ptask();
        if( pt != nullptr ) return pt;
        else return pt2.first_ptask();
      }

      inline ParallelTask* last_ptask() const
      {
        auto * pt = pt2.last_ptask();
        if( pt != nullptr ) return pt;
        else return pt1.last_ptask();
      }

      inline ParallelTaskQueue* ptask_queue() const { return pt1.ptask_queue(); }
      inline void attach_to_queue(ParallelTaskQueue* ptq) { pt1.attach_to_queue(ptq); pt2.attach_to_queue(ptq); }

      template<class T>
      inline auto operator >> ( T && rhs )
      {
        return ptask_sequence_after( std::move(*this) , std::move(rhs) );
      }

      template<class T>
      inline auto operator || ( T && rhs )
      {
        return ptask_schedule_after( std::move(*this) , std::move(rhs) );
      }

      inline void reset() { pt1.reset(); pt2.reset(); }
      inline void finalize()
      {
        if( pt1.ptask_queue() != nullptr )
        {
          if( pt2.ptask_queue() == nullptr ) { pt2.attach_to_queue( pt1.ptask_queue() ); }
          assert( pt1.ptask_queue() == pt2.ptask_queue() );
          pt1.finalize();
          pt2.finalize();
        }
      }
      inline void execute()
      {
        if( pt1.ptask_queue() != nullptr )
        {
          //std::cout<<"execute "; print(std::cout);
          PTaskBinExprConnector<Op,PTask1,PTask2>::connect( pt1 , pt2 ); 
          pt1.execute();
          pt2.execute();        
        }
      }

      inline ~PTaskBinExpr() { finalize(); execute(); reset(); }
      template<class S> S& print(S& out,int d=0)
      {
        out<<"(";
        pt1.print(out,d+1);
        switch(_Op)
        {
          case SEQUENCE_AFTER: out<<">>"; break;
          case SCHEDULE_AFTER: out<<"||"; break;
        }
        pt2.print(out,d+1);
        out<<")";
        if(d==0) out<<"\n";
        return out;
      }
    };

    template<class PTask1, class PTask2>
    static inline PTaskBinExpr<SEQUENCE_AFTER,PTask1,PTask2> ptask_sequence_after( PTask1 && t1 , PTask2 && t2 )
    {
      return { std::move(t1) , std::move(t2) };
    }

    template<class PTask1, class PTask2>
    static inline PTaskBinExpr<SCHEDULE_AFTER,PTask1,PTask2> ptask_schedule_after( PTask1 && t1 , PTask2 && t2 )
    {
      return { std::move(t1) , std::move(t2) };
    }

  }  
}

#if 0
    // ======================== Schedule merged parallel tasks ================================
    struct ParallelTaskEnqueueHelper
    {
      ParallelTaskQueue& ptq;
      ParallelTask* ptask = nullptr;
      ParallelTask* prec_ptask = nullptr;
      bool enqueue_task = true;
      bool immediate_flush = false;
      
      inline void reset() { ptask = nullptr; prec_ptask = nullptr; enqueue_task = true; immediate_flush = false; }
      
      // sequence after relation
      inline ParallelTaskEnqueueHelper operator >> ( ParallelTaskEnqueueHelper && ptte_suc )
      {
        auto cur_pt = ptask;
        auto suc_pt = ptte_suc.ptask;
        reset();
        ptte_suc.reset();
        suc_pt->sequence_after( cur_pt );
        // std::cout<<"sequence after : "<<suc_pt->m_tag<<" -> "<<cur_pt->m_tag<<std::endl;
        return { ptte_suc.ptq, suc_pt, cur_pt , false };
      }
      
      // schedule after relation
      inline ParallelTaskEnqueueHelper operator || ( ParallelTaskEnqueueHelper&& ptte_suc )
      {
        assert( &ptte_suc.ptq == &ptq );
        auto cur_pt = ptask;
        auto suc_pt = ptte_suc.ptask;
        reset();
        ptte_suc.reset();
        ptte_suc.ptask->schedule_after( ptask );
        //ptte_suc.ptask->fulfills( ptask );
        // std::cout<<"schedule after : "<<suc_pt->m_tag<<" -> "<<cur_pt->m_tag<<std::endl;
        if( cur_pt->detached_task() && suc_pt->fulfills_task() )
        {
          // std::cout<<"detach/fulfill pair : "<<suc_pt->m_tag<<" -> "<<cur_pt->m_tag<<std::endl;
          suc_pt->fulfills( cur_pt );
        }
        return { ptte_suc.ptq , suc_pt , cur_pt , false };
      }

      inline ParallelTaskEnqueueHelper& operator >> ( const flush_token_t& )
      {
        immediate_flush = true;
        return *this;
      }

      inline void enqueue()
      {
        if( ptask == nullptr ) return;
        if( prec_ptask != nullptr )
        {
          // std::cout<<"Enqueue prec_ptask "<<prec_ptask->m_tag<< " this="<<this<< std::endl;          
          ptq.enqueue_ptask(prec_ptask);
        }
        if( enqueue_task )
        {
          // std::cout<<"Enqueue ptask "<<ptask->m_tag<< " this="<<this<< std::endl;          
          ptq.enqueue_ptask(ptask);
        }
        if( immediate_flush )
        {
          // std::cout<<"flush queue"<< std::endl;          
          ptq.flush();
        }
        reset();
      }

      inline ~ParallelTaskEnqueueHelper()
      {
        enqueue();
      }

    };

#endif

    

