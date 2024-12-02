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

#include <onika/macro_utils.h>

#define XNB_CHUNK_NEIGHBORS_CS_CASE(CS) _XNB_CHUNK_NEIGHBORS_CS_CASE_##CS
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_1   case 1
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_2   case 2
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_4   case 4
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_8   case 8
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_16  case 16
#define _XNB_CHUNK_NEIGHBORS_CS_CASE_VARIMPL default

#define XNB_CHUNK_NEIGHBORS_CS_VAR(CS,VAR,IN)  _XNB_CHUNK_NEIGHBORS_CS_VAR_##CS (VAR,IN)
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_1(VAR,IN)   static constexpr onika::UIntConst<1> VAR = {}
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_2(VAR,IN)   static constexpr onika::UIntConst<2> VAR = {}
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_4(VAR,IN)   static constexpr onika::UIntConst<4> VAR = {}
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_8(VAR,IN)   static constexpr onika::UIntConst<8> VAR = {}
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_16(VAR,IN)  static constexpr onika::UIntConst<16> VAR = {}
#define _XNB_CHUNK_NEIGHBORS_CS_VAR_VARIMPL(VAR,IN) const unsigned int VAR = IN

#undef XNB_COMMA // don't care about multiple definitions
#define XNB_COMMA ,

#ifndef XNB_CHUNK_NEIGHBORS_CS_LIST
#warning XNB_CHUNK_NEIGHBORS_CS_LIST shoud have been defined from CMake. using specializations 1,VARIMPL
#define XNB_CHUNK_NEIGHBORS_CS_LIST 8 XNB_COMMA 4 XNB_COMMA 1
#endif

#define __XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( FUNC , ... ) EXPAND_WITH_FUNC_NOSEP( FUNC ,##__VA_ARGS__ )
#define _XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( FUNC , LIST ) __XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( FUNC , LIST )
#define XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( FUNC ) _XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( FUNC , XNB_CHUNK_NEIGHBORS_CS_LIST )

