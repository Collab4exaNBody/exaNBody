/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once


#ifdef ONIKA_HAVE_OPENMP_TOOLS
#include <omp-tools.h>
#endif

//#include <thread>
#include <mutex>
#include <unordered_map>
#include <iostream>

#include <onika/omp/ompt_task_timing.h>
#include <onika/omp/ompt_task_info.h>
#include <onika/memory/streaming_storage_pool.h>

// #define _ONIKA_PROFILE_TASK_ALLOCATOR 1

namespace onika
{
namespace omp
{

  struct OpenMPToolThreadContext;
 
  class OpenMPToolInterace
  {
  public:
    typedef void (*TaskEventCallback) ( const OpenMPToolTaskTiming& );
    typedef std::ostream& (*AppContextPrinter) ( void* , std::ostream& );
  
#ifdef ONIKA_HAVE_OPENMP_TOOLS
    static int tool_initialize ( ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t *tool_data );
    static void tool_finalize ( ompt_data_t *tool_data );
#endif

    // global tool activation / deactivation
    static void enable();
    static void disable();

    // give information about application context and task tag (kernel name or relevant information)
    static const char* set_explicit_task_tag(const char* tag);
    static void task_begin( OpenMPTaskInfo* tinfo );
    static void task_end( OpenMPTaskInfo* tinfo );

    // user defined functions and callbacks
    static void set_task_start_callback( TaskEventCallback cb );
    static void set_task_stop_callback( TaskEventCallback cb );
    static void set_app_ctx_printer( AppContextPrinter printer );

    // calling thread unique identifier
    static int32_t get_current_thread_id();

    //  debugging feature
    static std::mutex dbg_mesg_mutex;
    static std::atomic<uint64_t> dbg_mesg_enabled;
    static inline std::mutex& dbg_message_mutex() { return dbg_mesg_mutex; }
    static inline bool dbg_message_enabled() { return dbg_mesg_enabled.load(std::memory_order_consume); }
    static inline void enable_dbg_message() { dbg_mesg_enabled.store(true,std::memory_order_release); }
    static inline void disable_dbg_message() { dbg_mesg_enabled.store(false,std::memory_order_release); }
    static inline void enable_internal_dbg_message() { user_internal_dbg_message_enabled.store(true,std::memory_order_release); }
    static inline void disable_internal_dbg_message() { user_internal_dbg_message_enabled.store(false,std::memory_order_release); }

    static std::ostream& print_task_info(std::ostream& out, const OpenMPTaskInfo* tinfo);

    // low level inspection (use for debugging purposes only, reserved for internal usage otherwise)
    static size_t num_thread_ctx();
    static OpenMPToolThreadContext* thread_ctx();
    static OpenMPToolThreadContext* thread_ctx(size_t i);

  private:

    // task info allocation
    //using TaskInfoStorageUnit = memory::StreamingStorageUnit<4096,8>;
    using TaskInfoStoragePool = memory::StreamingStoragePool;
    static TaskInfoStoragePool task_info_allocator;    
    static OpenMPTaskInfo* alloc_task_info(size_t n=1 /*, const char* place="<null>" */);
    static void free_task_info(OpenMPTaskInfo* /*, const char* place="<null>" */ );
#   ifdef ONIKA_HAVE_OPENMP_TOOLS

    static const char* ompt_set_result_string( ompt_set_result_t r );

    // ************* event handling *****************
    static void callback_thread_begin(ompt_thread_t thread_type, ompt_data_t *thread_data);    
    static void callback_thread_end(ompt_data_t *thread_data);
    static void callback_task_create(ompt_data_t *encountering_task_data, const ompt_frame_t *encountering_task_frame, ompt_data_t *new_task_data, int flags, int has_dependences, const void *codeptr_ra);
    static void callback_task_schedule(ompt_data_t *prior_task_data, ompt_task_status_t prior_task_status, ompt_data_t *next_task_data);
    static void callback_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data, ompt_data_t *task_data, unsigned int actual_parallelism, unsigned int index, int flags);
    static void callback_parallel_begin( ompt_data_t *encountering_task_data, const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data, unsigned int requested_parallelism, int flags, const void *codeptr_ra );
    static void callback_parallel_end( ompt_data_t *parallel_data, ompt_data_t *encountering_task_data, int flags, const void *codeptr_ra );
    static void callback_work(ompt_work_t wstype, ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data, ompt_data_t *task_data, uint64_t count, const void *codeptr_ra);
    // ********************************************************

    static ompt_set_callback_t ompt_set_callback;
    static ompt_get_thread_data_t ompt_get_thread_data;
    static ompt_get_task_info_t ompt_get_task_info;
    static ompt_get_state_t ompt_get_state;

    static inline constexpr uint64_t thread_id_magic_number = uint64_t(0x0000FEED);
    
#   endif

    // per thread context info
    static inline constexpr size_t max_thread_count = 4096;
    static OpenMPToolThreadContext* thread_ctx_table[ max_thread_count ];
    static size_t thread_count;

    // initial task info
    static OpenMPTaskInfo initial_task_info;

    // user event callbacks    
    static TaskEventCallback user_task_start_callback;
    static TaskEventCallback user_task_stop_callback;
    static AppContextPrinter user_app_ctx_printer;
   
    // global tool activation flag
    static bool tool_activated;

    static std::atomic<uint64_t> user_internal_dbg_message_enabled;

#   ifdef _ONIKA_PROFILE_TASK_ALLOCATOR
    static std::atomic<uint64_t> task_alloc_count;
    static std::atomic<uint64_t> task_alloc_retry;
    static std::atomic<uint64_t> task_alloc_yield;
    static std::atomic<uint64_t> task_alloc_ctxsw;
#   endif

    friend struct OpenMPToolThreadContext;
  };

} }

inline std::ostream& operator << ( std::ostream& out , const onika::omp::OpenMPTaskInfo& tinfo )
{
  return onika::omp::OpenMPToolInterace::print_task_info(out,&tinfo);
}

// ******************* utility macros ********************
#ifdef ONIKA_HAVE_OPENMP_TOOLS

#define onika_ompt_declare_task_context(v) \
  ::onika::omp::OpenMPTaskInfo v{"sequential"}
  
#define onika_ompt_begin_task_context2(v,p) \
  v.app_ctx = p; \
  ::onika::omp::OpenMPToolInterace::task_begin(&v)

#define onika_ompt_end_task_context2(v,p) \
  assert( v.app_ctx == p ); \
  ::onika::omp::OpenMPToolInterace::task_end(&v)

#define onika_ompt_begin_task_context(p) \
  onika_ompt_declare_task_context(__onika_omp_task_info); \
  onika_ompt_begin_task_context2(__onika_omp_task_info,p)

#define onika_ompt_end_task_context(p) \
  onika_ompt_end_task_context2(__onika_omp_task_info,p)

#define onika_ompt_begin_task(t) \
  const char* __onika_omp_task_tag = (t); \
  ::onika::omp::OpenMPTaskInfo __onika_omp_task_info{__onika_omp_task_tag}; \
  ::onika::omp::OpenMPToolInterace::task_begin(&__onika_omp_task_info)

// this one is deprecated
#define onika_ompt_end_task(t) \
  assert( ((const void*)__onika_omp_task_tag) == ((const void*)(t)) ); \
  __onika_omp_task_info.tag = __onika_omp_task_tag;\
  ::onika::omp::OpenMPToolInterace::task_end(&__onika_omp_task_info)

#define onika_ompt_end_current_task() \
  __onika_omp_task_info.tag = __onika_omp_task_tag;\
  ::onika::omp::OpenMPToolInterace::task_end(&__onika_omp_task_info)

#define onika_ompt_push_explicit_task_tag(t) \
  auto __onika_tag_backup = ::onika::omp::OpenMPToolInterace::set_explicit_task_tag(t)

#define onika_ompt_pop_explicit_task_tag() \
  ::onika::omp::OpenMPToolInterace::set_explicit_task_tag(__onika_tag_backup)

#else
#define onika_ompt_declare_task_context(v)    do{}while(false)
#define onika_ompt_begin_task_context2(v,p)   do{}while(false)
#define onika_ompt_end_task_context2(v,p)     do{}while(false)
#define onika_ompt_begin_task_context(p)      do{}while(false)
#define onika_ompt_end_task_context(p)        do{}while(false)
#define onika_ompt_begin_task(t)              do{}while(false)
#define onika_ompt_end_task(t)                do{}while(false)
#define onika_ompt_end_current_task()         do{}while(false)
#define onika_ompt_push_explicit_task_tag(t)  do{}while(false)
#define onika_ompt_pop_explicit_task_tag()    do{}while(false)
#endif

#define ONIKA_DBG_MESG_LOCK for( struct { size_t i=::onika::omp::OpenMPToolInterace::dbg_message_enabled(); const std::lock_guard<std::mutex> __dbg_mesg_lock{::onika::omp::OpenMPToolInterace::dbg_message_mutex()}; } __dbg_mesg; __dbg_mesg.i; __dbg_mesg.i=false )


