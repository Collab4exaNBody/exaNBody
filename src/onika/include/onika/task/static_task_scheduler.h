#pragma once

#include <onika/task/static_depend_lambda.h>

namespace onika
{
  namespace task
  {

    // generate omp task from lambda, given compile time known dependences
    template<int... Is , class... T>
    static inline auto static_task_scheduler( const char* tag , std::tuple<T*...> && args , std::integer_sequence<int,Is...> )
    {
      return onika::task::OpenMPLambdaTaskScheduler< std::integer_sequence<int,Is...> , std::tuple<T*...> > { tag , args };
    }

  }
}
