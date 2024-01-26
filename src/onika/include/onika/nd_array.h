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

namespace onika
{
  template<class T, long ... Dims> struct nd_array;
  template<class T> struct nd_array<T> { T data; };
  template<class T, long I> struct nd_array<T,I> { T data[I]; };
  template<class T, long I, long J> struct nd_array<T,I,J> { T data[J][I]; };
  template<class T, long I, long J, long K> struct nd_array<T,I,J,K> { T data[K][J][I]; };
  template<class T, long I, long J, long K, long L> struct nd_array<T,I,J,K,L> { T data[L][K][J][I]; };
}

