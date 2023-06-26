#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor.h>

namespace onika
{

  namespace task
  {

    struct DagSchedulerOMPDep
    {

      template<class PTDapImpl>
      static inline void schedule_tasks( PTDapImpl * self , ParallelTask* pt )
      {
        size_t n = self->dep_graph_impl().number_of_items();
        if( self->m_task_reorder.empty() ) for(size_t i=0;i<n;i++) spawn_omp_task( self, pt , i );
        else for(size_t i=0;i<n;i++) spawn_omp_task( self, pt , self->m_task_reorder[i] );
      }

      template<class PTDapImpl>
      static inline void spawn_omp_task( PTDapImpl * self , ParallelTask* pt , size_t i )
      {        
        if constexpr ( PTDapImpl::SpanT::ndims >= 1 )
        {
          const void * in_deps[ ONIKA_OMP_MAX_DEPENDS ];
          size_t n_indeps = self->in_dependences( i , in_deps );
          //size_t node_dep_depth = self->m_dep_depth[i];
          void * out_dep = nullptr;
          size_t n_out_deps = 0;
          //if ( node_dep_depth < self->m_max_dep_depth )
          {
            out_dep = self->out_dependence(i);
            n_out_deps = 1;
          }
          auto c = self->task_coord_int(pt,i); // properly scaled with grainsize and shifted with lower_bound
          omp::DynamicDependDispatcher { n_indeps, in_deps, n_out_deps, &out_dep } . invoke ( self , pt , c );
        }
        if constexpr ( PTDapImpl::SpanT::ndims == 0 ){ std::abort(); }
      }

    };


  }
}

