#pragma once

#include <onika/lambda_tools.h>
#include <cstdlib>
#include <onika/oarray.h>

// extra OpenMP task clauses for dynamic depend tasks
#if defined(__clang__)
#define ONIKA_DYNAMIC_DEPEND_OMP_TASK_ATTRIBUTES /*mergeable untied*/
#else
#define ONIKA_DYNAMIC_DEPEND_OMP_TASK_ATTRIBUTES /*mergeable untied*/
#endif

namespace onika
{

  namespace task { struct TaskPoolItem; struct ParallelTask; struct ParallelTaskExecutor; }

  namespace omp
  {
    // forward declarations
    struct DynamicDependOpenMPDispatch
    {
      virtual inline void operator () (const void ** indeps, void ** inoutdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt ) const { std::abort(); }
      virtual inline void operator () (const void ** indeps, void ** inoutdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,1> c ) const { std::abort(); }
      virtual inline void operator () (const void ** indeps, void ** inoutdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,2> c ) const { std::abort(); }
      virtual inline void operator () (const void ** indeps, void ** inoutdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,3> c ) const { std::abort(); }
      virtual inline void operator () (const void ** indeps, void ** inoutdeps, task::TaskPoolItem* tpi ) const { std::abort(); }
    };
    
    extern const DynamicDependOpenMPDispatch* dynamic_depend_dispatch_table [] ;
    
    inline const DynamicDependOpenMPDispatch& dynamic_depend_dispatch(size_t n_indeps , size_t n_outdeps)
    {
      size_t n = n_indeps + n_outdeps;
      size_t idx = n_indeps + ( n*(n+1) ) / 2; // idx for ndeps,nindeps
      assert( n <= ONIKA_OMP_MAX_DEPENDS );
      return * dynamic_depend_dispatch_table[ idx ];      
    }

    struct DynamicDependDispatcher
    {
      size_t n_indeps = 0;
      const void ** indeps = nullptr;
      size_t n_outdeps = 0;
      void ** outdeps = nullptr;

      template<class... CallArgs>
      inline void invoke ( const CallArgs& ... args ) const
      {
        dynamic_depend_dispatch(n_indeps,n_outdeps) ( indeps, outdeps, args... );
      }

    };
    
  }

}
