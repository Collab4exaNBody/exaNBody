#include <onika/task/parallel_task_queue.h>

namespace onika
{
  namespace task
  {
    static inline void task_friendly_lock(omp_lock_t* lock)
    {
      while( ! omp_test_lock(lock) )
      {
#       pragma omp taskyield
      }
    }

    std::atomic<ParallelTaskQueue*> ParallelTaskQueue::s_global_ptask_queue = nullptr;

    ParallelTaskQueue& ParallelTaskQueue::global_ptask_queue()
    {
      if( s_global_ptask_queue.load( std::memory_order_acquire ) == nullptr )
      {
#       pragma omp critical(onika_ptask_global_queue)
        {
          if( s_global_ptask_queue.load( std::memory_order_acquire ) == nullptr )
          {
            s_global_ptask_queue.store( new ParallelTaskQueue() );
          }
        }
      }
      return *s_global_ptask_queue;
    }

    ParallelTaskQueue::ParallelTaskQueue()
    {
      omp_init_lock(&m_queue_lock);
    }

    void ParallelTaskQueue::enqueue_token( ParallelTaskQueue::ParallelTaskToken && t )
    {
      task_friendly_lock(&m_queue_lock);
      if( m_pending_ptasks.empty() || m_pending_ptasks.back().type!=FLUSH || t.type!=FLUSH )
      {
        m_pending_ptasks.push_back( t );
      }
      omp_unset_lock(&m_queue_lock);      
    }

    void ParallelTaskQueue::enqueue_ptask( ParallelTask* pt )
    {
      assert( pt != nullptr );
      
      if( m_immediate_execute )
      {
        pt->set_auto_free_allocator( & allocator() );
        pt->schedule( /* ptq=nullptr */ ); // queue less scheduling
      }
      else
      {
        enqueue_token( { SCHEDULE , pt } );
      }
      // ONIKA_DBG_MESG_LOCK { std::cout<<"enqueue PTask @"<<(void*)pt<<", tag="<<pt->get_tag() <<", fifo size = "<<m_pending_ptasks.size()<< std::endl; }
    }

    void ParallelTaskQueue::wait_all()
    {
      flush();

      bool queue_empty = false;
      do
      {
        task_friendly_lock(&m_queue_lock);
        queue_empty = m_scheduled_ptasks.empty();
        omp_unset_lock(&m_queue_lock);
        if( ! queue_empty )
        {
#         pragma omp taskyield
        }
      } while( ! queue_empty );
    }

    void ParallelTaskQueue::flush()
    {      
      enqueue_token( { FLUSH } );      
      // ONIKA_DBG_MESG_LOCK { std::cout<<"flush ptask queue, fifo size = "<<m_pending_ptasks.size()<< std::endl; }
    }

    
    void ParallelTaskQueue::start()
    {
      auto * self = this;
#     pragma omp task default(none) firstprivate(self) untied
      self->scheduler_loop();
    }

    void ParallelTaskQueue::scheduler_loop()
    {
      // ONIKA_DBG_MESG_LOCK { std::cout << "ParallelTaskQueue::scheduler_loop() start" << std::endl << std::flush; }

      bool stop_scheduler = false;
      while( ! stop_scheduler )
      {
      
        task_friendly_lock(&m_queue_lock);
        while( m_pending_ptasks.empty() )
        {
          omp_unset_lock(&m_queue_lock);
#         pragma omp taskyield
          task_friendly_lock(&m_queue_lock);
        }

        auto tok = m_pending_ptasks.front();
        m_pending_ptasks.pop_front();

        switch( tok.type )
        {
          case SCHEDULE :
            assert( tok.ptask != nullptr );
            //m_scheduled_ptasks.push_back(tok.ptask);
            if( tok.ptask->implicit_scheduling() )
            {
              // ONIKA_DBG_MESG_LOCK { std::cout<<"PTQ TOKEN schedule (skipped) : PTask @"<<(void*)(tok.ptask) <<", tag="<<tok.ptask->get_tag() <<", scheduled tasks = "<<m_scheduled_ptasks.size()<< std::endl<< std::flush; }
            }
            else
            {
              // ONIKA_DBG_MESG_LOCK { std::cout<<"PTQ TOKEN schedule : PTask @"<<(void*)(tok.ptask) <<", tag="<<tok.ptask->get_tag() <<", scheduled tasks = "<<m_scheduled_ptasks.size()<< std::endl<< std::flush; }
              tok.ptask->schedule( this );
            }
            break;
          case FLUSH :
            // ONIKA_DBG_MESG_LOCK { std::cout<<"PTQ TOKEN flush"<<std::endl<< std::flush; }
            break;
          case STOP :
            // ONIKA_DBG_MESG_LOCK { std::cout<<"PTQ TOKEN stop"<<std::endl<< std::flush; }
            stop_scheduler = true;
            break;
          default :
            // ONIKA_DBG_MESG_LOCK { std::cerr<<"PTQ TOKEN unexpected"<<std::endl<< std::flush; }
            break;
        }

        omp_unset_lock(&m_queue_lock);
      }

      // ONIKA_DBG_MESG_LOCK { std::cout << "ParallelTaskQueue::scheduler_loop() end" << std::endl << std::flush; }
    }
    
    void ParallelTaskQueue::stop()
    {
      enqueue_token( { STOP } );      
    }

  }
}


