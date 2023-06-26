#include <onika/task/parallel_task.h>
#include <onika/dag/dag_algorithm.h>
#include <onika/task/tag_utils.h>
#include <onika/task/parallel_task_queue.h>

#include <chrono>

#include <onika/omp/ompt_interface.h>

namespace onika
{
  namespace task
  {


    void ParallelTask::sequence_after(ParallelTask* ptask)
    {
      assert(ptask != nullptr );
      std::scoped_lock lock(m_mutex, ptask->m_mutex);
      
      assert( m_sequenced_before == nullptr );
      assert( ptask->m_sequence_after == nullptr );

      // ONIKA_DBG_MESG_LOCK { std::cout<<"link ptasks : " << ptask->get_tag() <<" >> "<< get_tag() << std::endl; }

      m_sequenced_before = ptask;      
      ptask->m_sequence_after = this;
    }

    void ParallelTask::schedule_after(ParallelTask* ptask)
    {
      assert(ptask != nullptr );
      std::scoped_lock lock(m_mutex, ptask->m_mutex);
      
      assert( ptask->m_schedule_after == nullptr );
      assert( m_scheduled_before == nullptr );

      // ONIKA_DBG_MESG_LOCK { std::cout<<"link ptasks : "<<ptask->get_tag() <<" || "<< get_tag() << std::endl; }
      
      m_scheduled_before = ptask;
      ptask->m_schedule_after = this;
    }

    void ParallelTask::fulfills(ParallelTask* ptask)
    {
      std::scoped_lock lock(m_mutex);
      m_fulfilled_ptask = ptask;
    }

    void ParallelTask::all_tasks_completed_notify()
    {
      if( ! m_mutex.try_lock() )
      {
        std::cerr<<"Internal error: ParallelTask::all_tasks_completed_notify called concurrently\n";
        std::abort();
      }
      
      // ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"]::all_tasks_completed_notify"<< std::endl; }
      
      m_ptexecutor->all_tasks_completed_notify();
      if( m_sequence_after != nullptr )
      {
        ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"] release sequenced-after dependency"<< std::endl; }
        omp_fulfill_event( m_end_of_ptask_event );
      }
      omp_completion_events().clear();
      
      // self destroy when completed and any dependent task has been notified
      m_mutex.unlock();
      if( m_auto_free_allocator != nullptr )
      {
        assert( m_ptq == nullptr );
        //std::cout<<"ParallelTask self destruct with m_auto_free_allocator : m_ptexecutor refcount = "<<m_ptexecutor.use_count() <<"\n";
        m_ptexecutor = nullptr;
        this->~ParallelTask();
        m_auto_free_allocator->free( this , sizeof(ParallelTask) );
      }
      else
      {
        assert( m_ptq != nullptr );
        //std::cout<<"ParallelTask self destruct with ptq->allocator() : m_ptexecutor refcount = "<<m_ptexecutor.use_count() <<"\n";
        auto ptq = m_ptq;
        m_ptq = nullptr;
        m_ptexecutor = nullptr;
        this->~ParallelTask();
        ptq->allocator().free( this , sizeof(ParallelTask) );
      }

    }

    template<size_t Nd>
    void ParallelTask::run_iteration_range(size_t start, size_t end)
    {
      assert( end >= start );      
      for(size_t i=start;i<end;i++)
      {
        auto c = span().template index_to_coord<Nd>(i);
        m_ptexecutor->execute( this , c );
      }
    }

    template void ParallelTask::run_iteration_range<1>(size_t start, size_t end);
    template void ParallelTask::run_iteration_range<2>(size_t start, size_t end);
    template void ParallelTask::run_iteration_range<3>(size_t start, size_t end);

    void ParallelTask::account_completed_task(ssize_t nr)
    {
      assert( nr >= 0 );
      if( nr > 0 )
      {
        auto r = m_n_running_tasks.fetch_sub( nr ,std::memory_order_release);
        assert( r > 0 );
        if( r == nr )
        {
          // ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"]::account_completed_task("<<nr<<") , stock="<<r<<std::endl; }
          all_tasks_completed_notify();
        }
      }
    }

    template<size_t Nd>
    void ParallelTask::completion_notify(const oarray_t<size_t,Nd>& c)
    {
      assert( m_detached );
      assert( span().inside(c) );
      size_t element_idx = span().coord_to_index( c );
      assert( span().template index_to_coord<Nd>(element_idx) == c );
      // _Pragma("omp critical(dbg_mesg)") std::cout<<"ParallelTask::completion_notify"<<format_array(c)<<std::endl;
      assert( element_idx < omp_completion_events().size() );
      omp_fulfill_event( omp_completion_events()[element_idx] );
      account_completed_task(1);      
    }

    template void ParallelTask::completion_notify(const oarray_t<size_t,1>& c);
    template void ParallelTask::completion_notify(const oarray_t<size_t,2>& c);
    template void ParallelTask::completion_notify(const oarray_t<size_t,3>& c);


    void ParallelTask::notify_completion_event_available(size_t task_index)
    {
    }

    /************************************** schedule parallel tasks ***********************************/
    void ParallelTask::build_dag()
    {
      assert( span().ndims >= 1 && span().ndims <= 3 );
      if( stencil().is_local() )
      {
        m_trivial_dag = true;
      }
      else
      {
#       ifdef ONIKA_HAVE_OPENMP_TOOLS
        onika_ompt_begin_task("onika-build-dag");
#       endif
        
        //auto T0 = std::chrono::high_resolution_clock::now();
        m_trivial_dag = ! m_ptexecutor->build_graph(this);
        
#       ifdef ONIKA_HAVE_OPENMP_TOOLS
        onika_ompt_end_current_task();
#       endif
      }
    }
    /***************************************************************************************************/


    /************************************** schedule parallel tasks ***********************************/
    
    /*
    Scheduling rules : 
      outdep generated if
        - DAG is used
        - detached requested
    */
    void ParallelTask::schedule( ParallelTaskQueue* ptq )
    {
      assert( span().ndims >= 0 && span().ndims <= 3 );
      // assert( ptq != nullptr );

      m_ptq = ptq;

      //ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"]::schedule( ptq@"<<(void*)ptq<<" )"<< std::endl; }

      if( m_sequence_after != nullptr )
      {
        // ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"] : end-of-execution detached dep-out("<<(void*)this<<")"<< std::endl; }
        ParallelTask* pt = this; if(pt==nullptr){ std::abort(); }
        omp_event_handle_t tmp_evt{};
        OMP_TASK_DETACH( /**/ , depend(out:pt[0]) , tmp_evt ) { /* empty task to create a dependency on ptask termination*/ }
        m_end_of_ptask_event = std::move(tmp_evt);
      }

      switch( span().ndims )
      {
        case 0: schedule_nd<0>(); break;
        case 1: schedule_nd<1>(); break;
        case 2: schedule_nd<2>(); break;
        case 3: schedule_nd<3>(); break;
      }
            
      if( m_schedule_after != nullptr )
      {
        // ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"] : chain schedule (sched-after) "<<m_schedule_after->get_tag()<< std::endl; }
        m_schedule_after->schedule( ptq );
      }
      if( m_sequence_after != nullptr )
      {
        // ONIKA_DBG_MESG_LOCK { std::cout<<"ParallelTask["<<get_tag()<<"] : chain schedule (seq-after) "<<m_sequence_after->get_tag()<< std::endl; }
        m_sequence_after->schedule( ptq );
      }
    }

/*
    // helper function, intel-19's icpc crashes if pragma is placed inside the "if constexpr"
    static inline void spawn_0d_with_prev_dep_helper( ParallelTask* _self , ParallelTask* _prev_pt )
    {
      ParallelTask* self = _self;
      ParallelTask* prev_pt = _prev_pt;
#     pragma omp task default(none) firstprivate(self) depend(in:prev_pt[0])
      {
        self->m_ptexecutor->execute( self );
        self->all_tasks_completed_notify();
      }
    }
*/

    template<size_t Nd>
    void ParallelTask::schedule_nd()
    {
      std::unique_lock<std::mutex> lk(m_mutex);
      assert( span().ndims == Nd );

      // 0-D special case
      if constexpr ( Nd == 0 )
      {
        m_num_elements = 1;
        m_num_tasks = 1;
        m_n_running_tasks.store( 1 , std::memory_order_release );
      }

      // generic n-D case (n>=1)
      if constexpr ( Nd > 0 )
      {
        build_dag();

        // number of elemnts to process
        m_num_elements = span().nb_cells();
        m_num_tasks =  span().nb_coarse_cells();
        //std::cout<<"ptask"<<get_tag()<<" : span="; m_span.to_stream(std::cout); std::cout<<", m_num_elements="<<m_num_elements<<", m_num_tasks="<<m_num_tasks<<std::endl;

        if( m_detached )
        {
          assert( m_num_elements == m_num_tasks );
          omp_completion_events().resize( m_num_tasks );
        }

        if( m_trivial_dag )
        {
          // in case of detached tasks, each task need two decrements : 1 after completion of task code and 1 after event is fulfilled
          size_t task_counter = m_detached ? (m_num_elements*2) : m_num_elements;
          m_n_running_tasks.store( task_counter , std::memory_order_release );
        }
        else
        {
          assert( ! m_detached );
          m_num_tasks = m_ptexecutor->dep_graph().number_of_items();
          /*
           FIXME: Warning: number of running tasks may be less than number of elements in span area, due to graph reduction
           this issue must be addressed when detached tasks are used.
          */
          // at least, account for skipped fine grained elements, even if we can't "detach" them by now
          size_t skipped_elements = m_ptexecutor->dag_skipped_elements();
          // std::cout<<"ptask"<<get_tag()<<", m_num_elements="<<m_num_elements<<", m_num_tasks="<<m_num_tasks<<", dag_span_elements="<<m_ptexecutor->dag_span_elements() <<std::endl;
          assert( m_ptexecutor->dag_span_elements() == m_num_elements );
          m_n_running_tasks.store( m_num_elements - skipped_elements , std::memory_order_release );
          assert( m_num_tasks == m_ptexecutor->dep_graph().number_of_items() ); 
        }
        
      } // end of n-D case (n>=1)
      
      lk.unlock();
      m_ptexecutor->spawn_omp_all_tasks( this );
    }

    template void ParallelTask::schedule_nd<1>();
    template void ParallelTask::schedule_nd<2>();
    template void ParallelTask::schedule_nd<3>();
    /***************************************************************************************************/



    /************************************** merge two parallel tasks ***********************************/
#   if 0
    void ParallelTask::merge(ParallelTask & pt)
    {
      assert( m_span.ndims >= 1 && m_span.ndims <= 3 );
      switch( m_span.ndims )
      {
        case 1: merge_nd<1>(pt); break;
        case 2: merge_nd<2>(pt); break;
        case 3: merge_nd<3>(pt); break;
      }
    }

    template<size_t Nd>
    void ParallelTask::merge_nd(ParallelTask & pt )
    {
      assert( m_span.ndims == Nd );
      assert( m_span.ndims == pt.m_span.ndims );
      assert( pt.m_data_count >= 1 && pt.m_data_count < MAX_ACCESSED_DATA_COUNT );
      assert( m_data_count >= 1 && m_data_count < MAX_ACCESSED_DATA_COUNT );
      assert( m_stencil.m_nbits == m_stencil_bits );
      assert( pt.m_stencil.m_nbits == pt.m_stencil_bits );

      // compute data ptrs shifts
      int data_idx[MAX_ACCESSED_DATA_COUNT];
      for(size_t i=0;i<pt.m_data_count;i++)
      {
        data_idx[i] = -1;
        for(size_t j=0;j<m_data_count;j++)
        {
          if( pt.m_data_ptr[i]==m_data_ptr[j] ) data_idx[i] = j;
        }
        if( data_idx[i] == -1 )
        {
          m_stencil_shift[m_data_count] = m_stencil_bits;
          m_data_ptr[m_data_count] = pt.m_data_ptr[i];
          data_idx[i] = m_data_count;
          ++ m_data_count;
          m_stencil_bits += pt.data_stencil_bits(i);
        }
      }

      // englarge resulting stencil bounding box as needed
      dac::AbstractStencil fusion_stencil;
      fusion_stencil.m_nbits = m_stencil_bits;
      fusion_stencil.m_ndims = m_stencil.m_ndims;        
      for(unsigned int k=0;k<m_stencil.m_ndims;k++)
      {
        fusion_stencil.m_low[k] = std::min( m_stencil.m_low[k] , pt.m_stencil.m_low[k] );
        fusion_stencil.m_size[k] = std::max( m_stencil.m_size[k] + m_stencil.m_low[k] , pt.m_stencil.m_size[k] + pt.m_stencil.m_low[k] ) - fusion_stencil.m_low[k];
      }
      fusion_stencil.clear_bits();

      // import bits from this->m_stencil
      const size_t o_n_cells = m_stencil.nb_cells();
      const auto o_size = m_stencil.box_size();
      const auto o_low = m_stencil.low_corner();
      const auto f_low = fusion_stencil.low_corner();
      const auto f_size = fusion_stencil.box_size();
      for(size_t i=0;i<o_n_cells;i++)
      {
        size_t j = coord_to_index( array_sub( array_add( index_to_coord(i,o_size) , o_low ) , f_low ) , f_size );
        fusion_stencil.add_ro_mask( m_stencil.ro_mask(i) , j );
        fusion_stencil.add_rw_mask( m_stencil.rw_mask(i) , j );
      }
      m_stencil = fusion_stencil;

      // import bits from pt.m_stencil
      const size_t p_n_cells = pt.m_stencil.nb_cells();
      const auto p_size = pt.m_stencil.box_size();
      const auto p_low = pt.m_stencil.low_corner();
      for(size_t i=0;i<p_n_cells;i++)
      {
        size_t j = coord_to_index( array_sub( array_add( index_to_coord(i,p_size) , p_low ) , f_low ) , f_size );
        for(size_t d=0;d<pt.m_data_count;d++)
        {
          size_t nb = pt.data_stencil_bits(d);
          assert( nb == data_stencil_bits(data_idx[d]) );
          for(size_t b=0;b<nb;b++)
          {
            m_stencil.add_bit( m_stencil_bits * (2*j  ) + m_stencil_shift[data_idx[d]] + b , pt.m_stencil.read_bit( pt.m_stencil_bits * (2*i  ) + pt.m_stencil_shift[d] + b ) );
            m_stencil.add_bit( m_stencil_bits * (2*j+1) + m_stencil_shift[data_idx[d]] + b , pt.m_stencil.read_bit( pt.m_stencil_bits * (2*i+1) + pt.m_stencil_shift[d] + b ) );
          }
        }
      }

      // merge spans
      m_span = dac::bounding_span( m_span , pt.m_span );
      m_task->chain_task( pt.m_task );
      pt.m_task = nullptr;
    }

    template void ParallelTask::merge_nd<1>(ParallelTask & pt );
    template void ParallelTask::merge_nd<2>(ParallelTask & pt );
    template void ParallelTask::merge_nd<3>(ParallelTask & pt );
#   endif
    
    
    /***************************************************************************************************/

    void omp_fullfil_callback( void* userData )
    {
      assert( userData != nullptr );
      omp_fulfill_event( * (omp_event_handle_t*) userData );
    }

    const char* ParallelTask::get_tag() const
    {
      return tag_filter_out_path( m_ptexecutor->tag() );
    }

  }
}


