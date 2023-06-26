#include <onika/task/task_pool_item.h>

namespace onika
{
  namespace task
  {
    std::atomic<int64_t> TaskPoolItem::s_stats_retry { 0 };
    std::atomic<int64_t> TaskPoolItem::s_stats_yield { 0 };
    std::atomic<int64_t> TaskPoolItem::s_stats_switch { 0 };
  }
}


