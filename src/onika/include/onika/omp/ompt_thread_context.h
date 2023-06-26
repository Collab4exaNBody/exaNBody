#pragma once

#include <onika/omp/ompt_task_timing.h>
#include <onika/omp/ompt_interface.h>
#include <onika/omp/ompt_task_info.h>
#include <utility>

namespace onika { namespace omp
{

  struct OpenMPToolThreadContext
  {  
    static constexpr ssize_t MAX_TASK_STACK_SIZE = 1024;
    ssize_t m_task_sp = -1;
    OpenMPToolTaskTiming m_task_stack[MAX_TASK_STACK_SIZE];

    inline ssize_t stack_size() const { return m_task_sp+1; }
    inline OpenMPToolTaskTiming& task_state() { return m_task_stack[ m_task_sp ]; }

    inline void notify_task_begin( void * ctx , const char* tag , std::chrono::nanoseconds start_time )
    {
      if( m_task_sp >= 0 )
      {
        task_state().end = start_time;
        trigger_task_stop( task_state() );
      }
      push( OpenMPToolTaskTiming{ ctx, tag, OpenMPToolInterace::get_current_thread_id(), start_time } );
      trigger_task_start( task_state() );
    }

    inline void notify_task_end( void* ctx , const char* tag , std::chrono::nanoseconds end_time )
    {
      int cur_cpuid = OpenMPToolInterace::get_current_thread_id();
      bool task_end_valid = ( m_task_sp >= 0 );
      if(task_end_valid) { task_end_valid = ( ctx == task_state().ctx && tag == task_state().tag && task_state().cpu_id == cur_cpuid ); }
      assert( task_end_valid );
      if( task_end_valid )
      {
        task_state().end = end_time;
        trigger_task_stop( task_state() );
      }
      if( m_task_sp >= 0 ) { pop(); }
      if( m_task_sp >= 0 )
      {
        task_state().cpu_id = cur_cpuid;
        task_state().timepoint = end_time;
        task_state().end = std::chrono::nanoseconds{0};
        ++ task_state().resume;
      }
    }

    inline void push( const OpenMPToolTaskTiming& t )
    {
      assert( m_task_sp < (ssize_t(MAX_TASK_STACK_SIZE)-1) );
      m_task_stack[ ++ m_task_sp ] = t;
    }

    inline void pop()
    {
      assert( m_task_sp >= 0 );
      -- m_task_sp;
    }

    // call user profile callbacks
    inline void trigger_task_start( const OpenMPToolTaskTiming& evt_info )
    {
      if( OpenMPToolInterace::user_task_start_callback != nullptr )
      {
        ( * OpenMPToolInterace::user_task_start_callback ) ( evt_info );
      }
    }
    inline void trigger_task_stop(const OpenMPToolTaskTiming& evt_info)
    {
      if( OpenMPToolInterace::user_task_stop_callback != nullptr )
      {
        ( * OpenMPToolInterace::user_task_stop_callback ) ( evt_info );
      }
    }

  };
  /******************************************************************/

} }

