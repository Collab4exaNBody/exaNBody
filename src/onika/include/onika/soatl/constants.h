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

#include <cstdlib> // for size_t

namespace onika { namespace soatl
{

  template<size_t N> struct Log2 { static constexpr size_t value = Log2<N/2>::value+1; };
  template<> struct Log2<1> { static constexpr size_t value = 0; };
  template<> struct Log2<0> { static constexpr size_t value = 0; };
  
  template<size_t n> struct IsPowerOf2 { static constexpr bool value = (n == (1ul<<Log2<n>::value)); };

  namespace cst
  {
	  template<size_t A> struct align
	  {
	    static_assert(A>0,"alignment must be > 0");
	    static_assert(IsPowerOf2<A>::value,"alignment must be a power of 2");
	    static constexpr size_t value = A;
	  };
	  
	  template<size_t C>
	  struct chunk
	  {
	    static_assert(C>0,"chunk size must be > 0");
	    static constexpr size_t value = C;
	  };
	  
	  template<size_t> struct at {};
	  template<size_t> struct count {};

	  static constexpr cst::align<1> unaligned;
	  static constexpr cst::align<1> aligned_1;
	  static constexpr cst::align<2> aligned_2;
	  static constexpr cst::align<4> aligned_4;
	  static constexpr cst::align<8> aligned_8;
	  static constexpr cst::align<16> aligned_16;
	  static constexpr cst::align<32> aligned_32;
	  static constexpr cst::align<64> aligned_64;

	  static constexpr cst::chunk<1> no_chunk;
	  static constexpr cst::chunk<1> chunk_1;
	  static constexpr cst::chunk<2> chunk_2;
	  static constexpr cst::chunk<4> chunk_4;
	  static constexpr cst::chunk<8> chunk_8;
	  static constexpr cst::chunk<16> chunk_16;
  }

} } // namespace soatl

