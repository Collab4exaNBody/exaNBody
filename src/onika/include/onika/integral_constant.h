#pragma once

#include <onika/cuda/cuda.h>

namespace onika
{

  template<class T, T _Value>
  struct IntegralConst
  {
    ONIKA_HOST_DEVICE_FUNC inline constexpr operator T () { return _Value; }
  };
  template<bool B> using BoolConst = IntegralConst<bool,B>;
  template<unsigned int I> using UIntConst = IntegralConst<unsigned int,I>;
  template<int I> using IntConst = IntegralConst<int,I>;
  template<unsigned int I> using UIntConst = IntegralConst<unsigned int,I>;

  using FalseType = BoolConst<false>;
  using TrueType = BoolConst<true>;
}


