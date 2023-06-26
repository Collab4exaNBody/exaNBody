#pragma once

#include <onika/oarray.h>
#include <cstdint>

namespace onika
{

  namespace task
  {

    struct ParallelTaskCostInfo
    {
      uint64_t cost = 0;
      int count = 0;
      int skipped = 0;
      inline bool operator != (const ParallelTaskCostInfo& rhs) const { return cost!=rhs.cost || count!=rhs.count || skipped!=rhs.skipped; }
    };

    template<size_t Nd>
    struct NullTaskCostFunc
    {
      inline constexpr uint64_t operator () ( oarray_t<size_t,Nd> ) const { return 1; }
    };

    template<class T> struct IsNullTaskCostFunc : public std::false_type {};
    template<size_t Nd> struct IsNullTaskCostFunc< NullTaskCostFunc<Nd> > : public std::true_type {};
    template<class T> static inline constexpr bool is_null_cost_func_v = IsNullTaskCostFunc<T>::value ;

    //static_assert(  );

  }

}

