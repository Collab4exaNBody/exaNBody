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
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <atomic>

#include <onika/omp/ompt_interface.h>
#include <onika/omp/ompt_thread_context.h>
#include <onika/omp/ompt_task_timing.h>
#include <onika/omp/tag_utils.h>
#include <onika/macro_utils.h>

//#include <unordered_map>
//#include <unordered_set>

namespace onika
{
namespace omp
{
  // user and internal debug messages locking and activation
  std::mutex OpenMPToolInterace::dbg_mesg_mutex;
  std::atomic<uint64_t> OpenMPToolInterace::dbg_mesg_enabled = true;
  std::atomic<uint64_t> OpenMPToolInterace::user_internal_dbg_message_enabled = false;

# ifndef NDEBUG
  static inline constexpr bool __onika_internal_dbg_mesg_enable = true;
# else
  static inline constexpr bool __onika_internal_dbg_mesg_enable = false;
# endif

// debug message macros
# define ONIKA_INT_DBG_MESG_LOCK if constexpr (__onika_internal_dbg_mesg_enable) if(::onika::omp::OpenMPToolInterace::user_internal_dbg_message_enabled.load(std::memory_order_consume)) \
  for( struct { size_t i=true; const std::lock_guard<std::mutex> __dbg_mesg_lock{::onika::omp::OpenMPToolInterace::dbg_mesg_mutex}; } __dbg_mesg; __dbg_mesg.i; __dbg_mesg.i=false )

# define ONIKA_ERR_MESG_LOCK for( struct { size_t i=true; const std::lock_guard<std::mutex> __dbg_mesg_lock{::onika::omp::OpenMPToolInterace::dbg_mesg_mutex}; } __dbg_mesg; __dbg_mesg.i; __dbg_mesg.i=false )

//  task info struct allocation statistics
# ifdef _ONIKA_PROFILE_TASK_ALLOCATOR
  std::atomic<uint64_t> OpenMPToolInterace::task_alloc_count=0;
  std::atomic<uint64_t> OpenMPToolInterace::task_alloc_retry=0;
  std::atomic<uint64_t> OpenMPToolInterace::task_alloc_yield=0;
  std::atomic<uint64_t> OpenMPToolInterace::task_alloc_ctxsw=0;
# endif


# ifdef ONIKA_HAVE_OPENMP_TOOLS
  // tool instance pointer
  struct OpenMPToolInteracePrivateData
  {
    ompt_start_tool_result_t* m_tool_data = nullptr;
    uint64_t m_some_tool_data;
  };

  // ompt API functions obtained through function lookup
  ompt_set_callback_t OpenMPToolInterace::ompt_set_callback = nullptr;
  ompt_get_thread_data_t OpenMPToolInterace::ompt_get_thread_data = nullptr;
  ompt_get_task_info_t OpenMPToolInterace::ompt_get_task_info = nullptr;
  ompt_get_state_t OpenMPToolInterace::ompt_get_state = nullptr;
# endif

  // thread dispatch table
  OpenMPToolThreadContext* OpenMPToolInterace::thread_ctx_table[ OpenMPToolInterace::max_thread_count ];
  size_t OpenMPToolInterace::thread_count = 0;

  // user callbacks
  OpenMPToolInterace::TaskEventCallback OpenMPToolInterace::user_task_start_callback = nullptr;
  OpenMPToolInterace::TaskEventCallback OpenMPToolInterace::user_task_stop_callback = nullptr;
  OpenMPToolInterace::AppContextPrinter OpenMPToolInterace::user_app_ctx_printer = nullptr;

  // global tool activation flag
  bool OpenMPToolInterace::tool_activated = false;

  // ****** task info allocation system *****
  OpenMPTaskInfo OpenMPToolInterace::initial_task_info;
  OpenMPToolInterace::TaskInfoStoragePool OpenMPToolInterace::task_info_allocator;
  // *********************************************

  /***** utility functions for easy debugging (short hashes for pointers) ******/
  // it's thread safe if a buffer is given, and it's thread UNsafe otherwise
  static const char* small_uint64_hash(uint64_t value, char* str=nullptr)
  {
    static char static_storage[8];
    uint8_t s[8];
    std::memcpy( s , &value , 8 );
    uint8_t cs = 0;
    for(int i=0;i<8;i++) cs = cs ^ s[i];
    int cs0 = cs >> 4;
    int cs1 = cs & 15;
    if( str == nullptr ) str = static_storage;
    str[0] = cs0<10 ? ('0'+cs0) : ('A'+cs0-10);
    str[1] = cs1<10 ? ('0'+cs1) : ('A'+cs1-10);
    str[2] = '\0';
    return str;
  }
  static const char* pointer_short_name(const void* ptr , char* str=nullptr)
  {
    if(ptr==nullptr) return "null";
    return small_uint64_hash( ((const uint8_t*)ptr) - ((const uint8_t*)nullptr) , str );
  }
/*
  static const char* ompt_data_short_name(ompt_data_t* data , char* str=nullptr)
  {
    if( data == nullptr ) return small_uint64_hash( 0 , str );
    else return small_uint64_hash( ((uint8_t*)(data->ptr)) - ((uint8_t*)nullptr) , str );
  }
*/
  std::ostream& OpenMPToolInterace::print_task_info(std::ostream& out, const OpenMPTaskInfo* tinfo)
  {
    out<<pointer_short_name(tinfo)<<"(ctx=";
    if(OpenMPToolInterace::user_app_ctx_printer != nullptr ) (*OpenMPToolInterace::user_app_ctx_printer)(tinfo->app_ctx,out) ;
    else out <<pointer_short_name(tinfo->app_ctx);
    out<<",tag='"<< tag_filter_out_path(tinfo->tag)<<"',th="<<pointer_short_name(tinfo->thread_ctx)<<")";
//#   ifndef NDEBUG
//    out<<" alloc_site="<<tinfo->allocation_call_site;
//#   endif
    return out;
  }
  /*****************************************************************************/

  OpenMPTaskInfo* OpenMPToolInterace::alloc_task_info(size_t n /*, const char* place */)
  {
    //return new OpenMPTaskInfo { nullptr , nullptr , nullptr , nullptr , nullptr , true };
    auto && [ptr,retry,yield,ctxsw] = task_info_allocator.allocate_nofail( sizeof(size_t) + sizeof(OpenMPTaskInfo) * n , false );
#   ifdef _ONIKA_PROFILE_TASK_ALLOCATOR
    OpenMPToolInterace::task_alloc_count.fetch_add(n,std::memory_order_relaxed);
    OpenMPToolInterace::task_alloc_retry.fetch_add(retry,std::memory_order_relaxed);
    OpenMPToolInterace::task_alloc_yield.fetch_add(yield,std::memory_order_relaxed);
    OpenMPToolInterace::task_alloc_ctxsw.fetch_add(ctxsw,std::memory_order_relaxed);
#   endif
    * reinterpret_cast<size_t*>(ptr) = n;
    OpenMPTaskInfo* tinfo = reinterpret_cast<OpenMPTaskInfo*>( ptr + sizeof(size_t) );
    for(size_t i=0;i<n;i++)
    {
      tinfo[i].tag = nullptr;
      tinfo[i].app_ctx = nullptr;
      tinfo[i].thread_ctx = nullptr;
      tinfo[i].explicit_task_tag = nullptr;
      tinfo[i].dyn_alloc = true;
      tinfo[i].allocated = true;
#   ifndef NDEBUG
      tinfo[i].magic = OpenMPTaskInfo::DYN_ALLOCATED_TASK;
#   endif
    }
    ONIKA_INT_DBG_MESG_LOCK
    {
      std::cout<<"OMPT: alloc task_info: n="<<n;
      for(size_t i=0;i<n;i++)
      {
        std::cout<<" tinfo["<<i<<"]@"<<pointer_short_name(&tinfo[i]);
      }
      std::cout<<std::endl;
    }
    return tinfo;
  }

  void OpenMPToolInterace::free_task_info(OpenMPTaskInfo* tinfo /*, const char* place*/ )
  {
    assert( tinfo != nullptr );
    assert( tinfo->dyn_alloc && tinfo->allocated );
    char* ptr = reinterpret_cast<char*>(tinfo) - sizeof(size_t);
    size_t n = * reinterpret_cast<size_t*>(ptr);
    for(size_t i=0;i<n;i++)
    {
      assert( tinfo[i].dyn_alloc && tinfo[i].allocated && tinfo[i].magic == OpenMPTaskInfo::DYN_ALLOCATED_TASK );
      tinfo[i].tag = nullptr;
      tinfo[i].app_ctx = nullptr;
      tinfo[i].thread_ctx = nullptr;
      tinfo[i].explicit_task_tag = nullptr;
      tinfo[i].allocated = false;
      tinfo[i].dyn_alloc = false;
//#   ifndef NDEBUG
//      tinfo[i].magic = OpenMPTaskInfo::DYN_ALLOCATED_TASK;
//#   endif
    }

    ONIKA_INT_DBG_MESG_LOCK
    {
      std::cout<<"OMPT: free task_info: n="<<n;
      for(size_t i=0;i<n;i++)
      {
        std::cout<<" tinfo["<<i<<"]@"<<pointer_short_name(&tinfo[i]);
      }
      std::cout<<std::endl;
    }

    TaskInfoStoragePool::free(ptr, sizeof(size_t) + sizeof(OpenMPTaskInfo) * n );
   //delete tinfo;
  }

  static inline void assert_task_valid(OpenMPTaskInfo* tinfo)
  {
    assert( tinfo->allocated );
    assert( tinfo->magic == OpenMPTaskInfo::DYN_ALLOCATED_TASK || tinfo->magic == OpenMPTaskInfo::INITIAL_TASK || tinfo->magic == OpenMPTaskInfo::TEMPORARY_TASK );
    assert( ( tinfo->dyn_alloc && tinfo->magic == OpenMPTaskInfo::DYN_ALLOCATED_TASK ) || ( ! tinfo->dyn_alloc && tinfo->magic != OpenMPTaskInfo::DYN_ALLOCATED_TASK ) );
  }

  void OpenMPToolInterace::enable()
  {    
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    static std::mutex local_mutex;

    for(size_t i=0;i<OpenMPToolInterace::max_thread_count;i++) OpenMPToolInterace::thread_ctx_table[i]=nullptr;
    std::atomic<uint64_t> thread_id_counter = 0;

    // first touch OpenMP threads
#   pragma omp parallel shared(thread_id_counter)
    {
      std::lock_guard<std::mutex> lock(local_mutex);
      assert( ompt_get_thread_data != nullptr );
      ompt_data_t * data = ompt_get_thread_data();
      assert( data != nullptr );
      uint64_t tid = thread_id_counter.fetch_add(1);
      data->value = ( thread_id_magic_number << 32 ) | tid;
      OpenMPToolInterace::thread_ctx_table[ tid ] = new OpenMPToolThreadContext {};
    }
    OpenMPToolInterace::thread_count = thread_id_counter.load();

    auto tid = get_current_thread_id();
    //std::cout <<"intial_task tid="<<tid <<std::endl;
    auto cur_thread_ctx = OpenMPToolInterace::thread_ctx_table[ tid ];
    
    // populate task_data with initial_task info
    auto wclock = OpenMPToolTaskTiming::wall_clock_time();
    ompt_data_t * task_data = nullptr;
    ompt_get_task_info( 0 ,nullptr , &task_data, nullptr, nullptr, nullptr );
    assert( task_data != nullptr );
    initial_task_info.tag = "initial_task";
    initial_task_info.app_ctx = nullptr;
    initial_task_info.thread_ctx = cur_thread_ctx;
    initial_task_info.dyn_alloc = false;
    initial_task_info.allocated = true;
    task_data->ptr = &initial_task_info;
    assert( initial_task_info.thread_ctx != nullptr );

#   ifndef NDEBUG
    initial_task_info.magic = OpenMPTaskInfo::INITIAL_TASK;
#   endif

    ONIKA_INT_DBG_MESG_LOCK { std::cout<<"OMPT: initial_task tinfo="<<initial_task_info<<std::endl; }
    initial_task_info.thread_ctx->notify_task_begin( initial_task_info.app_ctx, initial_task_info.tag, wclock );

    // register tool callbacks
#   define register_callback_t(name, type)                       \
    do{                                                           \
      type f_##name = & OpenMPToolInterace::callback_##name;                            \
      ompt_set_result_t r; \
      if( (r=OpenMPToolInterace::ompt_set_callback( ompt_callback_##name, (ompt_callback_t)f_##name )) == ompt_set_never) \
        std::cerr<<"OMPT: Could not register callback'"<<#name<<"'"<<std::endl; \
      /*else std::cout<<"OMPT: registered "<< #name <<" ("<<ompt_set_result_string(r)<<")"<<std::endl;*/ \
    }while(0)
#   define register_callback(name) register_callback_t( name , ompt_callback_##name##_t )

    register_callback( thread_begin );
    register_callback( thread_end );
    register_callback( parallel_begin );
    register_callback( parallel_end );
    register_callback( implicit_task );
    register_callback( task_create );
    register_callback( work );
    register_callback( task_schedule );

#   undef register_callback
#   undef register_callback_t

    ONIKA_INT_DBG_MESG_LOCK { std::cout<<"OMPT: tool initialized with "<<thread_id_counter.load()<<" threads"<<std::endl; }

#   endif

    // clear task info allocated space to 0
    for(size_t i=0;i<task_info_allocator.NbStorageUnits;i++)
    {
      std::memset(task_info_allocator.m_storage_units[i].m_buffer , 0 , task_info_allocator.m_storage_units[i].BufferSize );
    }

    OpenMPToolInterace::tool_activated = true;
  }
  
  void OpenMPToolInterace::disable()
  {
    OpenMPToolInterace::tool_activated = false;

#   ifdef ONIKA_HAVE_OPENMP_TOOLS

#   define unregister_callback(name)                       \
    do{                                                           \
      ompt_set_result_t r; \
      if( (r=OpenMPToolInterace::ompt_set_callback( ompt_callback_##name, (ompt_callback_t)nullptr )) == ompt_set_never) \
        std::cerr<<"OMPT: Could not unregister callback'"<<#name<<"'"<<std::endl; \
      /*else std::cout<<"OMPT: unregistered "<< #name <<" ("<<ompt_set_result_string(r)<<")"<<std::endl;*/ \
    }while(0)

    unregister_callback( thread_begin );
    unregister_callback( thread_end );
    unregister_callback( parallel_begin );
    unregister_callback( parallel_end );
    unregister_callback( implicit_task );
    unregister_callback( task_create );
    unregister_callback( work );
    unregister_callback( task_schedule );

#   undef register_callback

#   endif

#   ifdef _ONIKA_PROFILE_TASK_ALLOCATOR
    std::cout<<"OMPT: task alloc count : "<< task_alloc_count.load() << std::endl;
    std::cout<<"OMPT: task alloc retry : "<< task_alloc_retry.load() << std::endl;
    std::cout<<"OMPT: task alloc yield : "<< task_alloc_yield.load() << std::endl;
    std::cout<<"OMPT: task alloc ctxsw : "<< task_alloc_ctxsw.load() << std::endl;
    auto && [a,f] = task_info_allocator.memory_usage();
    std::cout<<"OMPT: task alloc leaks : " << (a-f)/sizeof(OpenMPTaskInfo) << std::endl;
/*
    const size_t allocsz = TaskInfoStorageUnit::allocation_size( sizeof(OpenMPTaskInfo) );
    for(size_t i=0;i<task_info_allocator.NbStorageUnits;i++)
    {
      for(size_t j=0;j<task_info_allocator.m_storage_units[i].BufferSize;j+=allocsz)
      {
        OpenMPTaskInfo* tinfo = reinterpret_cast<OpenMPTaskInfo*>( task_info_allocator.m_storage_units[i].m_buffer + j );
        if(tinfo->allocated)
        {
          std::cout<<"\tleak @"<<(void*)tinfo<<" "; print_task_info(tinfo); std::cout<<std::endl;
        }
      }
    }
*/
#   endif
  }

  void OpenMPToolInterace::set_task_start_callback( TaskEventCallback cb )
  {
    OpenMPToolInterace::user_task_start_callback = cb;
  }
  
  void OpenMPToolInterace::set_task_stop_callback( TaskEventCallback cb )
  {
     OpenMPToolInterace::user_task_stop_callback = cb;
  }

  void OpenMPToolInterace::set_app_ctx_printer( AppContextPrinter printer )
  {
     OpenMPToolInterace::user_app_ctx_printer = printer;
  }

  int32_t OpenMPToolInterace::get_current_thread_id()
  {
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    ompt_data_t * data = ompt_get_thread_data();
    if( data != nullptr )
    {
      uint64_t id = data->value;
      if( (id>>32) == thread_id_magic_number )
      {
        id = id & ((1ull<<32)-1);
        return id;
      }
    }
#   endif
    return -1;
  }

  size_t OpenMPToolInterace::num_thread_ctx()
  {
    return OpenMPToolInterace::thread_count;
  }
  
  OpenMPToolThreadContext* OpenMPToolInterace::thread_ctx(size_t tid)
  {
    assert( tid >=0 && tid < thread_count );
    assert( thread_ctx_table[tid] != nullptr );
    return thread_ctx_table[tid];
  }

  OpenMPToolThreadContext* OpenMPToolInterace::thread_ctx()
  {
    if( ! OpenMPToolInterace::tool_activated ) return nullptr;
    auto tid = get_current_thread_id();
    assert( tid >=0 && size_t(tid) < thread_count );
    assert( thread_ctx_table[tid] != nullptr );
    return thread_ctx_table[tid];
  }

  const char* OpenMPToolInterace::set_explicit_task_tag(const char* tag)
  {
    if( ! OpenMPToolInterace::tool_activated ) return nullptr;
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    ompt_data_t * task_data = nullptr;
    ompt_get_task_info( 0 ,nullptr , &task_data, nullptr, nullptr, nullptr );
    assert( task_data != nullptr );
    OpenMPTaskInfo* cur = reinterpret_cast<OpenMPTaskInfo*>( task_data->ptr );
    assert( cur != nullptr );
    auto prev_tag = cur->explicit_task_tag;
    cur->explicit_task_tag = tag;
    ONIKA_INT_DBG_MESG_LOCK { std::cout<<"OMPT: set_explicit_task_tag "<< ( (tag!=nullptr) ? tag : "null" ) <<" tinfo="<< *cur <<std::endl; }
    return prev_tag;
#   else
    return nullptr;
#   endif  
  }

  void OpenMPToolInterace::task_begin( OpenMPTaskInfo* tinfo )
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    ompt_data_t * task_data = nullptr;
    ompt_get_task_info( 0 ,nullptr , &task_data, nullptr, nullptr, nullptr );
    assert( task_data != nullptr );

    OpenMPTaskInfo* cur = reinterpret_cast<OpenMPTaskInfo*>( task_data->ptr );
    auto cur_thread_ctx = OpenMPToolInterace::thread_ctx();
    auto wclock = OpenMPToolTaskTiming::wall_clock_time();

    tinfo->thread_ctx = cur_thread_ctx;
    tinfo->prev = nullptr; // prev is set only in case of nested application contexts, otherwise it must be nullptr

    ONIKA_INT_DBG_MESG_LOCK {
      std::cout<<"OMPT: task_begin prev="<<(*cur)<<" new="<<(*tinfo)<<std::endl;
    }

    if( cur!=nullptr ) // several begin/end in a single task or different nested app_ctx begin/end
    {
      assert( cur != tinfo );
      if( tinfo->app_ctx == nullptr )
      {
        tinfo->app_ctx = cur->app_ctx;
      }
      // a null tag indicate a non tracked task
      if( cur->tag!=nullptr )
      {
        // cur is a nesting task or application context, it must be stacked and carefuly sopped/resumed
        assert( cur->thread_ctx = cur_thread_ctx );    
        cur->thread_ctx->notify_task_end( cur->app_ctx, cur->tag, wclock );
      }
      tinfo->prev = cur;
    }
    assert( tinfo->app_ctx != nullptr );
    assert( tinfo->tag != nullptr );
    tinfo->thread_ctx->notify_task_begin( tinfo->app_ctx, tinfo->tag, OpenMPToolTaskTiming::wall_clock_time() );
    task_data->ptr = tinfo;
#   endif
  }

  void OpenMPToolInterace::task_end( OpenMPTaskInfo* tinfo )
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    assert( tinfo != nullptr );

    ompt_data_t * task_data = nullptr;
    ompt_get_task_info( 0 ,nullptr , &task_data, nullptr, nullptr, nullptr );
    assert( task_data != nullptr );
    
    assert( reinterpret_cast<OpenMPTaskInfo*>( task_data->ptr ) == tinfo );
    auto cur_thread_ctx = OpenMPToolInterace::thread_ctx();
    assert( cur_thread_ctx != nullptr );
    assert( cur_thread_ctx == tinfo->thread_ctx );
    
    ONIKA_INT_DBG_MESG_LOCK {
      std::cout<<"OMPT: task_end cur="<<(*tinfo)
      <<" prev="<<pointer_short_name(tinfo->prev)<<" tag="<<tinfo->tag<<" th="<<pointer_short_name(tinfo->thread_ctx)
      <<" curth="<<pointer_short_name(cur_thread_ctx)<<std::endl;
    }

    auto wclock = OpenMPToolTaskTiming::wall_clock_time();

    assert( tinfo->thread_ctx != nullptr );
    tinfo->thread_ctx->notify_task_end( tinfo->app_ctx, tinfo->tag, wclock );
    
    if( tinfo->prev != nullptr )
    {
      tinfo->prev->thread_ctx = cur_thread_ctx;
      if( tinfo->prev->tag != nullptr )
      {
        cur_thread_ctx->notify_task_begin( tinfo->prev->app_ctx, tinfo->prev->tag, wclock );
      }
    }
    task_data->ptr = tinfo->prev;
#   endif    
  }

# ifdef ONIKA_HAVE_OPENMP_TOOLS
  const char* OpenMPToolInterace::ompt_set_result_string( ompt_set_result_t r )
  {
    switch( r )
    {
      case ompt_set_error: return "ompt_set_error"; break;
      case ompt_set_never: return "ompt_set_never"; break;
      case ompt_set_impossible: return "ompt_set_impossible"; break;
      case ompt_set_sometimes: return "ompt_set_sometimes"; break;
      case ompt_set_sometimes_paired: return "ompt_set_sometimes_paired"; break;
      case ompt_set_always: return "ompt_set_always"; break;
      default: return "<unknown>"; break;
    }
  }

  int OpenMPToolInterace::tool_initialize( ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t *tool_data )
  {    
    //std::cout << "OMPT: initialize"<<std::endl;
    OpenMPToolInterace::ompt_set_callback = ( ompt_set_callback_t ) ( (*lookup) ("ompt_set_callback") );
    OpenMPToolInterace::ompt_get_thread_data = ( ompt_get_thread_data_t ) ( (*lookup) ("ompt_get_thread_data") );
    OpenMPToolInterace::ompt_get_task_info = ( ompt_get_task_info_t ) ( (*lookup) ("ompt_get_task_info") );
    OpenMPToolInterace::ompt_get_state = ( ompt_get_state_t ) ( (*lookup) ("ompt_get_state") );
    return 1;
  }

  void OpenMPToolInterace::tool_finalize ( ompt_data_t *tool_data )
  {
    //std::cout << "OMPT: finalize"<<std::endl;
    OpenMPToolInteracePrivateData* priv = reinterpret_cast<OpenMPToolInteracePrivateData*>( tool_data->ptr );
    delete priv->m_tool_data;
    delete priv;
  }

  void OpenMPToolInterace::callback_thread_begin(ompt_thread_t thread_type, ompt_data_t *thread_data)
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
//    std::cout << "OMPT: thread_begin"<<std::endl;
  }

  void OpenMPToolInterace::callback_thread_end(ompt_data_t *thread_data)
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
//    std::cout << "OMPT: thread_end"<<std::endl;
  }

  void OpenMPToolInterace::callback_task_create(ompt_data_t *encountering_task_data, const ompt_frame_t *encountering_task_frame, ompt_data_t *new_task_data, int flags, int has_dependences, const void *codeptr_ra)
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
    assert( encountering_task_data != nullptr && new_task_data != nullptr );
    const char* flags_str[10];
    int fc = 0;
    if( flags & ompt_task_initial )    flags_str[fc++]="initial";
    if( flags & ompt_task_implicit )   flags_str[fc++]="implicit";
    if( flags & ompt_task_explicit )   flags_str[fc++]="explicit";
    if( flags & ompt_task_target )     flags_str[fc++]="target";
    if( flags & ompt_task_undeferred ) flags_str[fc++]="undeferred";
    if( flags & ompt_task_untied )     flags_str[fc++]="untied";
    if( flags & ompt_task_final )      flags_str[fc++]="final";
    if( flags & ompt_task_mergeable )  flags_str[fc++]="mergeable";
    if( flags & ompt_task_merged )     flags_str[fc++]="merged";

    new_task_data->ptr = nullptr;
    if( flags & ompt_task_explicit )
    {
      OpenMPTaskInfo* enc_tinfo = reinterpret_cast<OpenMPTaskInfo*>( encountering_task_data->ptr );
      if( enc_tinfo != nullptr )
      {
        auto tinfo = alloc_task_info();
        tinfo->app_ctx = enc_tinfo->app_ctx;        
        tinfo->tag = enc_tinfo->explicit_task_tag;
        new_task_data->ptr = tinfo;

        auto cur_thread_ctx = OpenMPToolInterace::thread_ctx();
        ONIKA_INT_DBG_MESG_LOCK {
          std::cout<<"OMPT: create task @"<<(void*)tinfo; for(int i=0;i<fc;i++) std::cout<<" "<<flags_str[i];
          std::cout<<" curthread="<<pointer_short_name(cur_thread_ctx)<<" tinfo="<<(*tinfo)<<" enc="<<(*enc_tinfo)<<std::endl;
        }

      }
    }
  }

  void OpenMPToolInterace::callback_task_schedule(ompt_data_t *prior_task_data, ompt_task_status_t prior_task_status, ompt_data_t *next_task_data)
  {
    if( ! OpenMPToolInterace::tool_activated ) return;

    // debugger friendly early cast
    OpenMPTaskInfo* prior = nullptr;
    if( prior_task_data != nullptr) { prior = reinterpret_cast<OpenMPTaskInfo*>( prior_task_data->ptr ); }
    OpenMPTaskInfo* next = nullptr;
    if( next_task_data != nullptr) { next = reinterpret_cast<OpenMPTaskInfo*>( next_task_data->ptr ); }

    const char* schedule_point = "<unknow>";

    /*
    Scheduling notes :
      - detach : switch to the execution of another task without completing the task. a late_fulfill event will follow on the 'terminated-but-not-completed' task
      - late_fulfill : a task previously terminated task is allowed to complete after a fulfill event has been received
      - early_fulfill : a task still executing received a fulfill event. thus it won't detach when done, instead a 'complete' scheduling event will switch its executing thread to execution of another task
    */

    switch( prior_task_status )
    {
      case ompt_task_complete: schedule_point="complete"; break;
      case ompt_task_yield: schedule_point="yield"; break;
      case ompt_task_cancel: schedule_point="cancel"; break;
      case ompt_task_detach: schedule_point="detach"; break;
      case ompt_task_early_fulfill: schedule_point="early_fulfill"; break;
      case ompt_task_late_fulfill: schedule_point="late_fulfill"; break;
      case ompt_task_switch: schedule_point="switch"; break;
      /*case ompt_taskwait_complete*/ default : schedule_point="taskwait_complete/other"; break;
    }

    //assert( prior_task_data != nullptr && next_task_data != nullptr );

    auto cur_thread_ctx = OpenMPToolInterace::thread_ctx();
    assert( cur_thread_ctx != nullptr );
    
    //if( prior_task_status == ompt_task_yield )
    ONIKA_INT_DBG_MESG_LOCK
    {
      std::cout<<"OMPT: schedule "<<schedule_point<<" curth="<<pointer_short_name(cur_thread_ctx)<<" prior=";
      if( prior != nullptr ) { std::cout<<(*prior); } else { std::cout<<"<null>"; }
      std::cout<<" next=";
      if( next != nullptr ) { std::cout<<(*next); } else { std::cout<<"<null>"; }
      std::cout<<std::endl;
    }

    auto wclock = OpenMPToolTaskTiming::wall_clock_time();
    if( prior != nullptr ) 
    {
      assert_task_valid( prior );
      if( prior->thread_ctx != nullptr && prior->tag != nullptr )
      {
        prior->thread_ctx->notify_task_end( prior->app_ctx, prior->tag, wclock );
      }
      prior->thread_ctx = nullptr; // not attached to a thread anymore
      if( prior_task_status == ompt_task_complete || prior_task_status == ompt_task_cancel || prior_task_status == ompt_task_late_fulfill )
      {
        prior->tag = nullptr;
        if( prior->dyn_alloc ) { free_task_info( prior ); }
        prior_task_data->ptr = nullptr;
      }
    }
    if( next != nullptr )
    {
      assert( next->allocated );
      next->thread_ctx = cur_thread_ctx;
      if( next->tag != nullptr )
      {
        next->thread_ctx->notify_task_begin( next->app_ctx, next->tag, wclock );
      }
    }

  }

  void OpenMPToolInterace::callback_parallel_begin( ompt_data_t *encountering_task_data, const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data, unsigned int requested_parallelism, int flags, const void *codeptr_ra )
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
    assert(parallel_data!=nullptr && encountering_task_data!=nullptr);

    ONIKA_INT_DBG_MESG_LOCK 
    {
      OpenMPTaskInfo* enc = reinterpret_cast<OpenMPTaskInfo*>( encountering_task_data->ptr );
      assert( enc != nullptr );
      std::cout<<"OMPT: parallel_begin NP="<<requested_parallelism <<" enc="<<(*enc)<<std::endl;
    }

    OpenMPTaskInfo* enc = reinterpret_cast<OpenMPTaskInfo*>( encountering_task_data->ptr ); 
    if( enc != nullptr )
    {
      OpenMPTaskInfo* tinfo = alloc_task_info(requested_parallelism);
      for(unsigned int i=0;i<requested_parallelism;i++)
      {
        tinfo[i].app_ctx = enc->app_ctx;
        tinfo[i].tag = nullptr; // not accounted
        tinfo[i].thread_ctx = nullptr; // not accounted
        tinfo[i].explicit_task_tag = enc->explicit_task_tag;
      }
      parallel_data->ptr = tinfo;
    }
    else { parallel_data->ptr = nullptr; }
    //parallel_data->ptr = encountering_task_data->ptr;
  }
  
  void OpenMPToolInterace::callback_parallel_end( ompt_data_t *parallel_data, ompt_data_t *encountering_task_data, int flags, const void *codeptr_ra )
  {
    if( ! OpenMPToolInterace::tool_activated ) return;
    assert(parallel_data!=nullptr && encountering_task_data!=nullptr);
    
    ONIKA_INT_DBG_MESG_LOCK 
    {
      std::cout<<"OMPT: parallel_end ";
      OpenMPTaskInfo* enc = reinterpret_cast<OpenMPTaskInfo*>( encountering_task_data->ptr );
      if( enc != nullptr )
      {
        std::cout<<" enc="<<(*enc);
      }
      std::cout<<std::endl;
    }

    OpenMPTaskInfo* tinfo = reinterpret_cast<OpenMPTaskInfo*>( parallel_data->ptr );
    if( tinfo != nullptr )
    {
      free_task_info(tinfo);
      parallel_data->ptr = nullptr;
    }
  }

  void OpenMPToolInterace::callback_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data, ompt_data_t *task_data, unsigned int actual_parallelism, unsigned int index, int flags)
  {
    if( ! OpenMPToolInterace::tool_activated ) return;

    assert( task_data != nullptr );

    switch( endpoint )
    {
      case ompt_scope_begin :
      {
        task_data->ptr = nullptr;
        OpenMPTaskInfo* tinfo = nullptr;
        if( parallel_data != nullptr ) tinfo = reinterpret_cast<OpenMPTaskInfo*>( parallel_data->ptr );
        if( tinfo != nullptr )
        {
          //tinfo[index].app_ctx = enc->app_ctx;
          tinfo[index].tag = nullptr; // not accounted
          tinfo[index].thread_ctx = nullptr; // not accounted
          task_data->ptr = & tinfo[index];
          ONIKA_INT_DBG_MESG_LOCK 
          {
            std::cout<<"OMPT: implicit task begin P"<<index<<'/'<<actual_parallelism<<" @"<<(void*)(&tinfo[index])<<" tinfo="<<pointer_short_name(&tinfo[index])<<std::endl;
          }
        }
      }
      break;
      
      case ompt_scope_end :
      {
        OpenMPTaskInfo* tinfo = reinterpret_cast<OpenMPTaskInfo*>( task_data->ptr );
        if( tinfo != nullptr )
        {
          ONIKA_INT_DBG_MESG_LOCK 
          {
            std::cout<<"OMPT: implicit task end P"<<index<<'/'<<actual_parallelism<<" @"<<(void*)(&tinfo[index])<<" tinfo="<<pointer_short_name(&tinfo[index])<<std::endl;
          }
        }
        task_data->ptr = nullptr;
      }
      break;

      /*case ompt_scope_beginend:*/
      default : { std::cerr<<"ompt_scope_beginend/unknown ("<< ((int)endpoint) <<") endpoint not handled"<<std::endl; std::abort(); } break;
    }


  }

  void OpenMPToolInterace::callback_work(ompt_work_t wstype, ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data, ompt_data_t *task_data, uint64_t count, const void *codeptr_ra)
  {
    static const char* wstype_str [] = {
      "<unknown>" ,
      "loop",
      "sections", 
      "single_executor", 
      "single_other", 
      "workshare", 
      "distribute", 
      "taskloop", 
      "scope" };

    if( ! OpenMPToolInterace::tool_activated ) return;
    assert(parallel_data!=nullptr && task_data!=nullptr);
    
    auto wclock = OpenMPToolTaskTiming::wall_clock_time();
    OpenMPTaskInfo* cur_task = reinterpret_cast<OpenMPTaskInfo*>( task_data->ptr );
    
    if( cur_task == nullptr )
    {
      ONIKA_INT_DBG_MESG_LOCK { std::cout<<"OMPT: work "<< ((endpoint==ompt_scope_begin)?"begin ":"end ") << wstype_str[wstype]  <<" null-task"<<std::endl; }
      return;
    }
    
    auto cur_thread_ctx = OpenMPToolInterace::thread_ctx();
    switch( endpoint )
    {
      case ompt_scope_begin :
      {
        cur_task->tag = wstype_str[wstype];
        cur_task->thread_ctx = cur_thread_ctx;
        if( wstype == ompt_work_taskloop && cur_task->explicit_task_tag == nullptr ) { cur_task->explicit_task_tag = wstype_str[wstype]; }

        ONIKA_INT_DBG_MESG_LOCK 
        {
          std::cout<<"OMPT: work begin task="<<(*cur_task)<<std::endl;
        }

        cur_task->thread_ctx->notify_task_begin( cur_task->app_ctx, cur_task->tag, wclock );
      }
      break;
      
      case ompt_scope_end :
      {
        assert( cur_task->tag == wstype_str[wstype] );
        assert( cur_task->thread_ctx == cur_thread_ctx );

        ONIKA_INT_DBG_MESG_LOCK 
        {
          std::cout<<"OMPT: work end tinfo="<<(*cur_task)<<std::endl;
        }

        cur_task->thread_ctx->notify_task_end( cur_task->app_ctx, cur_task->tag, wclock );
        cur_task->tag = nullptr;
        cur_task->thread_ctx = nullptr;
        if( wstype == ompt_work_taskloop && cur_task->explicit_task_tag == wstype_str[wstype] ) { cur_task->explicit_task_tag = nullptr; }
      }
      break;

      /*case ompt_scope_beginend :*/
      default : { std::cerr<<"ompt_scope_beginend/unknown ("<< ((int)endpoint) <<") endpoint not handled"<<std::endl; std::abort(); } break;
    }
  }

# endif // ONIKA_HAVE_OPENMP_TOOLS

} } // namespace onika::omp



#ifdef ONIKA_HAVE_OPENMP_TOOLS

extern "C"
{
  ompt_start_tool_result_t *ompt_start_tool( unsigned int omp_version, const char *runtime_version );
}

ompt_start_tool_result_t *ompt_start_tool( unsigned int omp_version, const char *runtime_version )
{
//  std::cout<<"OMPT: VER="<<omp_version<<" RT="<<runtime_version<<std::endl<<std::flush;
  auto* priv = new onika::omp::OpenMPToolInteracePrivateData { nullptr , 0 };
  auto* tool = new ompt_start_tool_result_t { onika::omp::OpenMPToolInterace::tool_initialize , onika::omp::OpenMPToolInterace::tool_finalize , { .ptr = priv } };
  priv->m_tool_data = tool;
  return tool;
}

#endif

