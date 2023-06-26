#pragma once

#include <onika/task/task_pool_item.h>

#include <type_traits>

namespace onika
{
  namespace task
  {
  
    template<class Dag, class TaskFunc, class PointerMapFunc>
    struct TaskGroup
    {
      Dag m_dag;
      TaskFunc m_func;
      PointerMapFunc m_element_pointer;
      
      template<class Allocator>
      inline void generate_tasks( Allocator& alloc )
      {
        const size_t nb_items = m_dag.number_of_items();
        const auto * taskf = & m_func;

        for(size_t i=0;i<nb_items;i++)
        {
          const void* in_deps[ONIKA_OMP_MAX_DEPENDS];
          size_t n_indeps = 0;
          void* out_dep[1];
          size_t n_outdeps = 0;
          
          const auto coord = m_dag.item_coord(i);
          //if( m_dag.item_out_dep(i) ) { out_dep[0] = m_element_pointer( coord ); n_outdeps = 1; }
          for(auto d:m_dag.item_deps(i)) { in_deps[n_indeps++] = m_element_pointer( d ); }

//          omp::DynamicDependLambdaTask launcher{ n_indeps, in_deps, n_outdeps, out_dep };
          TaskPoolItem::lambda( alloc , [taskf,coord]()->void{(*taskf)(coord);} , n_indeps, in_deps, n_outdeps, out_dep )->omp_task();
//          auto task_func_adapter = [=]() { (*taskf) ( coord ); };
//          launcher << task_func_adapter;
        }
      }
      
    };
     
  }
}
