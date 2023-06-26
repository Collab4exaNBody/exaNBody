#pragma once

#include <cstdlib>
#include <omp.h>

namespace onika
{

  namespace task
  {

    static inline constexpr int ONIKA_DAG_SCHEDULER_AUTO = -1;
    static inline constexpr int ONIKA_DAG_SCHEDULER_OMPDEP = 0;
    static inline constexpr int ONIKA_DAG_SCHEDULER_NATIVEORDER = 1;
    static inline constexpr int ONIKA_DAG_SCHEDULER_NATIVEFIFO = 2;
    static inline constexpr int ONIKA_DAG_SCHEDULER_GPU = 3;

    struct ParallelTaskConfig
    {    
      // ---------- static members ------------------
      static int s_dag_scheduler;
      static bool s_dag_reduce;
      static bool s_dag_reorder;
      static bool s_dag_diagnostics;
      static bool s_dag_scheduler_tied;
      static bool s_dag_scheduler_yield;
      static double s_dag_graph_mt; // graph construction multi-threading : [0;1[ means fraction of cores to use as num of threads, integers 1...n means number of threads

      static double s_dag_bulk_task_factor; 
      static int s_dag_tasks_per_thread; 
      static int s_dag_max_batch_size;

      static int s_gpu_sm_mult;
      static int s_gpu_sm_add;
      static int s_gpu_block_size;

      static inline bool dag_reduce() { return s_dag_reduce; }
      static inline bool dag_reorder() { return s_dag_reorder; }
      static inline double dag_diagnostics() { return s_dag_diagnostics; }
      static inline int dag_scheduler() { return s_dag_scheduler; }
      static inline bool dag_scheduler_tied() { return s_dag_scheduler_tied; }
      static inline bool dag_scheduler_yield() { return s_dag_scheduler_yield; }
      static inline int dag_graph_mt() { return (s_dag_graph_mt < 1.0) ? static_cast<int>(omp_get_num_threads()*s_dag_graph_mt) : static_cast<int>(s_dag_graph_mt); }

      static inline double dag_bulk_task_factor()   { return s_dag_bulk_task_factor; }
      static inline int dag_tasks_per_thread()   { return s_dag_tasks_per_thread; }
      static inline double dag_max_batch_size()   { return s_dag_max_batch_size; }

      static inline int gpu_sm_mult() { return s_gpu_sm_mult; }
      static inline int gpu_sm_add() { return  s_gpu_sm_add; }
      static inline int gpu_block_size() { return  s_gpu_block_size; }

    };

  }
}


