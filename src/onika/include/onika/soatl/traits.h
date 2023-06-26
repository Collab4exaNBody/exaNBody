#pragma once

#include <type_traits>

namespace onika
{
  namespace soatl
  {

    template<class T> struct IsFieldArrays  : public std::false_type {};
    template<class T> static inline constexpr bool is_field_arrays_v = IsFieldArrays<T>::value;
    
  }
}

