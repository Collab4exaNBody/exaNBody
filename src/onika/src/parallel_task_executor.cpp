#include <onika/task/parallel_task_executor.h>

namespace onika
{
  namespace task
  {
    std::mutex ParallelTaskExecutor::s_ptexecutor_cache_mutex;
    std::unordered_map< std::size_t , std::shared_ptr<ParallelTaskExecutor> > ParallelTaskExecutor::s_ptexecutor_cache;

    void ParallelTaskExecutor::clear_ptexecutor_cache()
    {
      std::scoped_lock lock(s_ptexecutor_cache_mutex);
      s_ptexecutor_cache.clear();
    }

  }
}


