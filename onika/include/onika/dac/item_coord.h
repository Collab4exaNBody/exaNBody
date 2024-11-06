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

#include <cstdint>
#include <onika/oarray.h>

namespace onika
{

  namespace dac
  {

    template<size_t Nd> struct ItemCoordTypeHelper { using item_coord_t = oarray_t<size_t,Nd>; static inline constexpr item_coord_t zero = {}; };
    template<> struct ItemCoordTypeHelper<1> { using item_coord_t = oarray_t<size_t,1>; static inline constexpr item_coord_t zero = {0}; };
    template<> struct ItemCoordTypeHelper<2> { using item_coord_t = oarray_t<size_t,2>; static inline constexpr item_coord_t zero = {0,0}; };
    template<> struct ItemCoordTypeHelper<3> { using item_coord_t = oarray_t<size_t,3>; static inline constexpr item_coord_t zero = {0,0,0}; };
    template<> struct ItemCoordTypeHelper<4> { using item_coord_t = oarray_t<size_t,4>; static inline constexpr item_coord_t zero = {0,0,0,0}; };

    template<size_t Nd> using item_nd_coord_t = typename ItemCoordTypeHelper<Nd>::item_coord_t;

  }
  
}


