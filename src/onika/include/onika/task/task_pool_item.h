#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>

#include <onika/lambda_tools.h>
#include <onika/omp/dynamic_depend_dispatch.h>
#include <onika/memory/streaming_storage_unit.h>
#include <onika/memory/streaming_storage_pool.h>

// #define ONIKA_TASK_POOL_ITEM_STATS 1

namespace onika
{
  namespace task
  {
    
    struct TaskPoolItem
    {
      struct EmbeddedDeps
      {
        uint32_t n_indeps = 0;
        uint32_t n_outdeps = 0;
        void* deps[0];
      };

      static_assert( sizeof(EmbeddedDeps) == 2*sizeof(uint32_t) , "EmbeddedDeps not packed as expected" );
      static_assert( sizeof(char) == 1 , "expected char to be a single byte" );

      static inline constexpr uint32_t ASYNC_EXEC = 0x01;
      static inline constexpr uint32_t EMBEDDED_DEPS = 0x02;
    
      using TaskAllocator = memory::StreamingStoragePool;

      //TaskPoolStorageUnit* container = nullptr;
      const void* call_address = nullptr;
      uint32_t call_data_size = 0;
      uint32_t flags = 0; // if false, execution should not be deferred (i.e. using if(0) clause when generating omp task)
      TaskPoolItem* chained_task = nullptr;
#     ifndef NDEBUG
      size_t call_args_hash = 0;
#     endif
      char data[0];
      
      inline bool async_exec() const { return ( flags & ASYNC_EXEC ) != 0 ; }
      inline void set_async_exec( bool b )
      {
        if( b ) flags |= ASYNC_EXEC;
        else flags &= ~ASYNC_EXEC;
      }
      
      /* optional embedded dependences */
      inline bool has_embedded_deps() const { return ( flags & EMBEDDED_DEPS ) != 0 ; }
      inline uint32_t embedded_in_dep_count() const
      {
        if( has_embedded_deps() )
        {
          const EmbeddedDeps* deps = reinterpret_cast<const EmbeddedDeps*>( data + call_data_size );
          return deps->n_indeps;
        }
        else return 0;
      }
      inline const void** embedded_in_deps() const
      {
        if( has_embedded_deps() )
        {
          const EmbeddedDeps* deps = reinterpret_cast<const EmbeddedDeps*>( data + call_data_size );
          return (const void**) & deps->deps[0];
        }
        else return nullptr;
      }
      inline uint32_t embedded_out_dep_count() const
      {
        if( has_embedded_deps() )
        {
          const EmbeddedDeps* deps = reinterpret_cast<const EmbeddedDeps*>( data + call_data_size );
          return deps->n_outdeps;
        }
        else return 0;
      }
      inline void** embedded_out_deps() const
      {
        if( has_embedded_deps() )
        {
          const EmbeddedDeps* deps = reinterpret_cast<const EmbeddedDeps*>( data + call_data_size );
          return (void**)( ( & deps->deps[0] ) + deps->n_indeps );
        }
        else return nullptr;
      }
      
      
      static inline size_t needed_storage( size_t call_data_size , uint32_t ni=0, uint32_t no=0)
      {
        size_t dpsz = 0;
        if( (ni+no) > 0 )
        {
          dpsz = sizeof(EmbeddedDeps) + (ni+no) * sizeof(void*);
        }
        return sizeof(TaskPoolItem) + call_data_size + dpsz;
      }
      
      inline size_t storage_size() const
      {
        size_t sz = sizeof(TaskPoolItem) + call_data_size;
        if( has_embedded_deps() )
        {
          const EmbeddedDeps* deps = reinterpret_cast<const EmbeddedDeps*>( data + call_data_size );
          sz += sizeof(EmbeddedDeps) + ( deps->n_indeps + deps->n_outdeps ) * sizeof(void*) ;
        }
        return sz;
      }

      template<class F>
      static inline TaskPoolItem* lambda( TaskAllocator& allocator , F && f , uint32_t n_indeps=0, const void** indeps=nullptr, uint32_t n_outdeps=0, void** outdeps=nullptr)
      {      
        auto && [ ci_call_addr , ci_call_data , ci_data_size ] = lambda_call_info(f);
        size_t needed_storage = TaskPoolItem::needed_storage( ci_data_size , n_indeps , n_outdeps );
        
        auto && [p,nr,ny,sw] = allocator.allocate_nofail( needed_storage );
        assert( p != nullptr );

#       ifdef ONIKA_TASK_POOL_ITEM_STATS
        s_stats_retry.fetch_add( nr , std::memory_order_relaxed );
        s_stats_yield.fetch_add( ny , std::memory_order_relaxed );
        s_stats_switch.fetch_add( sw , std::memory_order_relaxed );
#       endif

        TaskPoolItem* item = reinterpret_cast<TaskPoolItem*>( p );
        assert( ( p + sizeof(TaskPoolItem) ) == item->data );
        item->call_address = ci_call_addr;
        item->call_data_size = ci_data_size;        
        item->flags = ASYNC_EXEC;
        item->chained_task = nullptr;

#       ifndef NDEBUG
        item->call_args_hash = lambda_call_args_hash(f);
#       endif

        std::memcpy( item->data , ci_call_data , ci_data_size );
        
        // optionally copy embedded dependences
        if( (n_indeps+n_outdeps) > 0 )
        {
          item->flags |= EMBEDDED_DEPS;
          EmbeddedDeps* deps = reinterpret_cast<EmbeddedDeps*>( item->data + item->call_data_size );
          deps->n_indeps = n_indeps;
          deps->n_outdeps = n_outdeps;
          void ** d = deps->deps;
          for(uint32_t i=0;i<n_indeps;i++) *(d++) = (void*) indeps[i];
          for(uint32_t i=0;i<n_outdeps;i++) *(d++) = outdeps[i];
        }

        assert( item->storage_size() == needed_storage );
        
        return item;
      }

      inline void chain_task(TaskPoolItem* next)
      {
        auto * ti = this;
        while( ti->chained_task != nullptr ) ti = ti->chained_task;
        ti->chained_task = next;
      }


      template<class... Args>
      inline void spawn_omp_task_auto_free()
      {
        assert( call_args_hash == FunctionCallArgsHash< std::function<void()> >::value() );
        omp::DynamicDependDispatcher { embedded_in_dep_count(), embedded_in_deps(), embedded_out_dep_count(), embedded_out_deps() } . invoke ( this );
      }

      /*
      template<class... Args>
      inline void spawn_omp_task(const Args & ... args)
      {
        static_assert( ( ... && (!std::is_same_v<Args,std::nullptr_t>) ) , "std::nullptr_t call argument is forbidden" );
        static_assert( ( ... && (!std::is_same_v<Args,TaskPoolItem*>) ) , "TaskPoolItem* call argument is forbidden" );
        assert( call_args_hash == FunctionCallArgsHash< std::function<void(Args...)> >::value() );
        omp::DynamicDependDispatcher { embedded_in_dep_count(), embedded_in_deps(), embedded_out_dep_count(), embedded_out_deps() } . invoke( this , args ... );
      }

      template<class... Args>
      inline void spawn_omp_task_deps(size_t n_indeps, const void ** indeps, size_t n_outdeps, void ** outdeps, const Args & ... args)
      {
        static_assert( ( ... && (!std::is_same_v<Args,std::nullptr_t>) ) , "std::nullptr_t call argument is forbidden" );
        static_assert( ( ... && (!std::is_same_v<Args,TaskPoolItem*>) ) , "TaskPoolItem* call argument is forbidden" );
        assert( call_args_hash == FunctionCallArgsHash< std::function<void(Args...)> >::value() );
        assert( ! has_embedded_deps() );
        omp::DynamicDependDispatcher { n_indeps, indeps, n_outdeps, outdeps } . invoke( this , args ... );
      }
      */

      template<class... Args>
      inline void execute(Args... args)
      {     
        static_assert( ( ... && (!std::is_reference_v<Args>) ) , "unexpected reference type in parameter list" );
        assert( call_args_hash == FunctionCallArgsHash< std::function<void(Args...)> >::value() );
        onika::lambda_call( call_address , data , args... );
        if(chained_task!=nullptr)
        {
          chained_task->execute( args... );
        }
      }

      inline void free()
      {
        if( chained_task != nullptr )
        {
          chained_task->free();
          chained_task = nullptr;
        }
        size_t s = storage_size();
        call_address = nullptr;
        call_data_size = 0;
        TaskAllocator::free( this , s );
      }

      static std::atomic<int64_t> s_stats_retry;
      static std::atomic<int64_t> s_stats_yield;
      static std::atomic<int64_t> s_stats_switch;
    };

    static_assert( sizeof(TaskPoolItem) == ( sizeof(void*) + sizeof(uint32_t)*2 ) + sizeof(TaskPoolItem*)
#       ifndef NDEBUG
        + sizeof(size_t)
#       endif
      , "TaskPoolItem not packed as expected" );   
  }
}

//#include <iostream>

/*
#define XSTAMP_OMP_LAMDA_SCHEDULE_DBG \
_Pragma("omp critical(dbg_mesg)") \
{ \
  if(ndeps>=0){ std::cout<<std::dec<<"sched("<<task->call_address<<","<<(void*)(task->data)<<")"; \
  if(nindeps>0){ std::cout<<" IN"; for(int zz=0;zz<nindeps;zz++) std::cout<<" "<< *(void**)(task->data+task->call_data_size+zz*sizeof(void*)) ; } \
  if((ndeps-nindeps)>0){ std::cout<<" INOUT"; for(int zz=0;zz<(ndeps-nindeps);zz++) std::cout<<" "<< *(void**)(task->data+task->call_data_size+(zz+nindeps)*sizeof(void*)); } \
  std::cout<<" async="<<std::boolalpha<<task->async_exec<<std::endl; }\
}
*/


/*
#define XSTAMP_OMP_LAMDA_EXECUTE_DBG \
_Pragma("omp critical(dbg_mesg)") \
{ std::cout<<std::dec<<"exec("<<task->call_address<<","<<(void*)(task->data)<<")"<<std::endl; }
*/
