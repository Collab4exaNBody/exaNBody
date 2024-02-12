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
#include <vector>
#include <onika/flat_tuple.h> // for tuple_index_t definition

  // =========== data access control constants ===========
namespace onika
{
  namespace dac
  {
    // central element access type
    //using ro_t = stencil_t<ssize_t,0, oarray_t< std::pair<ssize_t,bool>,0>{} >
    struct ro_t {};
    static inline constexpr ro_t ro{};
    
    struct rw_t {};
    static inline constexpr rw_t rw{};
 
    // single slice for a non sliceable data
    struct whole_t {};
    static inline constexpr whole_t whole;

    // slices for a std::pair
    using pair_first_t = tuple_index_t<0>;
    static inline constexpr pair_first_t pair_first{};
    
    using pair_second_t = tuple_index_t<1>;
    static inline constexpr pair_second_t pair_second{};

  }
  // ========================================================
}

