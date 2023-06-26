#pragma once

#include <onika/dac/dac.h>
#include <onika/dac/item_coord.h>
#include <onika/dac/box_span.h>
#include <onika/oarray_stream.h>
#include <onika/omp/ompt_interface.h>

#include <onika/task/parallel_task_cost.h>
#include <onika/task/parallel_task_executor.h>
#include <onika/stream_utils.h>
#include <omp.h>

#include <cstdlib>
#include <algorithm>

namespace onika
{

  namespace task
  {
    class ParallelTaskQueue;

    enum ParallelTaskFlags
    {
      PTASK_FLAG_DETACHED = 0x0001,
      PTASK_FLAG_FULFILL  = 0x0002,
      PTASK_FLAG_OUT_DEPS = 0x0004
    };

    struct ParallelTask
    {
      using ParallelTaskAllocator = memory::StreamingStoragePool;

      // ---------- instance members ------------------

      //const char* m_tag = nullptr;
            
      // detach / fulfill
      // std::vector<omp_event_handle_t> m_omp_completion_events; // => m_dtdap->m_omp_completion_events
      ParallelTask* m_fulfilled_ptask = nullptr;

      size_t m_num_elements = 0; // number of elements in span space
      size_t m_num_tasks = 0;    // number of distinct tasks scheduled
      
      // inter parallel task synchronization mechanisms
      std::mutex m_mutex;
      std::atomic<ssize_t> m_n_running_tasks = 0;
      ParallelTask* m_sequenced_before = nullptr;
      ParallelTask* m_sequence_after = nullptr;
      omp_event_handle_t m_end_of_ptask_event;

      // parallel tasks to be scheduled after this one is scheduled (all its tasks are scheduled)
      ParallelTask* m_scheduled_before = nullptr;
      ParallelTask* m_schedule_after = nullptr;

      // scheduling DAG
      std::shared_ptr<ParallelTaskExecutor> m_ptexecutor = nullptr;

      // auto self destruct upon task completion
      ParallelTaskAllocator* m_auto_free_allocator = nullptr;
      
      // keep track of queue this ptask is scheduled in
      ParallelTaskQueue* m_ptq = nullptr;

      // omp scheduling parameters
      bool m_detached = false;
      bool m_fulfill = false;
      bool m_gen_omp_out_dep = false;
      bool m_trivial_dag = false;

      // ---------- implementation -------------------

      inline ParallelTask( uint64_t flags /* , const char* tag */ )
      : /* m_tag( tag )
      , */ m_detached( ( flags & PTASK_FLAG_DETACHED ) != 0 )
      , m_fulfill( ( flags & PTASK_FLAG_FULFILL ) != 0 )
      , m_gen_omp_out_dep( ( flags & PTASK_FLAG_OUT_DEPS ) != 0 )
      {
        //std::cout<<"Construct ParallelTask @"<<(void*)this<<"\n";
      }
      
      /*inline ~ParallelTask()
      {
        std::cout<<"~ParallelTask @"<<(void*)this<<"\n";
      }*/

      inline void set_auto_free_allocator( ParallelTaskAllocator* a ) { m_auto_free_allocator = a; }

      inline const dac::abstract_box_span_t& span() const { return m_ptexecutor->span(); }
      inline const dac::AbstractStencil& stencil() const { return m_ptexecutor->stencil(); }
      inline std::vector<omp_event_handle_t>& omp_completion_events() { return m_ptexecutor->omp_completion_events(); }

      template<size_t Nd> void task_completed( const oarray_t<size_t,Nd>& );

      void sequence_after(ParallelTask* ptask);
      void schedule_after(ParallelTask* ptask);
      void fulfills(ParallelTask* ptask);

      inline bool implicit_scheduling() const { return m_scheduled_before!=nullptr || m_sequenced_before!=nullptr; }
 
      inline bool detached_task() const { return m_detached; }
      inline bool fulfills_task() const { return m_fulfill; }
 
      void schedule( ParallelTaskQueue* ptq = nullptr );
      template<size_t Nd> void schedule_nd();

#     if 0
      void merge(ParallelTask & pt);
      template<size_t Nd> void merge_nd(ParallelTask & pt );
#     endif
      
      void build_dag();
            
      template<size_t Nd> void run_iteration_range(size_t start, size_t end);
      void account_completed_task(ssize_t nr);
      void all_tasks_completed_notify();
      template<size_t Nd> void completion_notify(const oarray_t<size_t,Nd>& c);

      void notify_completion_event_available(size_t task_index);

      inline bool is_mergeable( const ParallelTask & pt ) const
      {
        return span().ndims == pt.span().ndims;
      }

      // may be used as a cuda callback. userData is interpredted as being a omp_event_handle_t*
      static void omp_fullfil_callback( void* userData );

      // get calling code mark string
      const char* get_tag() const;
     };

    // undefined or unused accessor context
    struct NullAccessorContext {};

    // access execution context in a running parallel task
    //template<class _FulfilledAccCtx = NullAccessorContext>
    struct ParallelTaskExecutionContext
    {
      ParallelTask * const m_pt = nullptr; // fulfilled or detached task ?
      //const _FulfilledAccCtx& m_fulfilled_ctx;
      //inline const _FulfilledAccCtx& fulfilled_ctx() { return m_fulfilled_ctx; }
      
      template<size_t Nd=0> inline void fulfill( oarray_t<size_t,Nd> c = oarray_t<size_t,Nd>{} ) const { m_pt->completion_notify(c); }
      inline const dac::abstract_box_span_t& fulfill_span() const { return m_pt->span(); }
    };

  }
}


