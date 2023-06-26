#include <onika/task/parallel_task_config.h>

namespace onika
{
  namespace task
  {
    double ParallelTaskConfig::s_dag_graph_mt = 0.75;

    double ParallelTaskConfig::s_dag_bulk_task_factor = 1.;
    int ParallelTaskConfig::s_dag_tasks_per_thread = 64;
    int ParallelTaskConfig::s_dag_max_batch_size = 16;

    int ParallelTaskConfig::s_dag_scheduler = ONIKA_DAG_SCHEDULER_AUTO;
    bool ParallelTaskConfig::s_dag_scheduler_tied = true;
    bool ParallelTaskConfig::s_dag_scheduler_yield = false;

    bool ParallelTaskConfig::s_dag_reduce = true;
    bool ParallelTaskConfig::s_dag_reorder = true;
    bool ParallelTaskConfig::s_dag_diagnostics = false;

    int ParallelTaskConfig::s_gpu_sm_mult = 4;
    int ParallelTaskConfig::s_gpu_sm_add = 0;
    int ParallelTaskConfig::s_gpu_block_size = 64;

  }
}


