/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <onika/cuda/cuda.h>

namespace onika
{

  template<class T, T _Value>
  struct IntegralConst
  {
    ONIKA_HOST_DEVICE_FUNC inline constexpr operator T () { return _Value; }
    ONIKA_HOST_DEVICE_FUNC inline bool operator == ( T other ) const { return _Value == other; }
    ONIKA_HOST_DEVICE_FUNC inline bool operator != ( T other ) const { return _Value != other; }
  };
  template<bool B> using BoolConst = IntegralConst<bool,B>;
  template<unsigned int I> using UIntConst = IntegralConst<unsigned int,I>;
  template<int I> using IntConst = IntegralConst<int,I>;
  template<unsigned int I> using UIntConst = IntegralConst<unsigned int,I>;

  using FalseType = BoolConst<false>;
  using TrueType = BoolConst<true>;
}


