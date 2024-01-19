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

/***************************** prefix variable args macros *************************************************/
// expand with prefix macro
#define EXPAND_WITH_PREFIX(...)               _EXPAND_WITH_PREFIX_HELPER1(0,_EXPAND_WITH_PREFIX_COUNT(__VA_ARGS__),##__VA_ARGS__)
#define EXPAND_WITH_PREFIX_PREPEND_COMMA(...) _EXPAND_WITH_PREFIX_HELPER1(1,_EXPAND_WITH_PREFIX_COUNT(__VA_ARGS__),##__VA_ARGS__)
#define _EXPAND_WITH_PREFIX_HELPER1(C,N,pfx,...) _EXPAND_WITH_PREFIX_HELPER2(C,N,pfx,##__VA_ARGS__)
#define _EXPAND_WITH_PREFIX_HELPER2(C,N,pfx,...) _EXPAND_WITH_PREFIX_HELPER3(C,N,pfx,##__VA_ARGS__)
#define _EXPAND_WITH_PREFIX_HELPER3(C,N,pfx,...) _EXPAND_WITH_PREFIX_##N(C,pfx,##__VA_ARGS__)
#define _EXPAND_WITH_PREFIX_1(C,pfx)
#define _EXPAND_WITH_PREFIX_2(C,pfx,a1)                              _EXPAND_WITH_PREFIX_START##C pfx a1
#define _EXPAND_WITH_PREFIX_3(C,pfx,a1,a2)                           _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2
#define _EXPAND_WITH_PREFIX_4(C,pfx,a1,a2,a3)                        _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3
#define _EXPAND_WITH_PREFIX_5(C,pfx,a1,a2,a3,a4)                     _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4
#define _EXPAND_WITH_PREFIX_6(C,pfx,a1,a2,a3,a4,a5)                  _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5
#define _EXPAND_WITH_PREFIX_7(C,pfx,a1,a2,a3,a4,a5,a6)               _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5 , pfx a6
#define _EXPAND_WITH_PREFIX_8(C,pfx,a1,a2,a3,a4,a5,a6,a7)            _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5 , pfx a6 , pfx a7
#define _EXPAND_WITH_PREFIX_9(C,pfx,a1,a2,a3,a4,a5,a6,a7,a8)         _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5 , pfx a6 , pfx a7 , pfx a8
#define _EXPAND_WITH_PREFIX_10(C,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9)     _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5 , pfx a6 , pfx a7 , pfx a8 , pfx a9
#define _EXPAND_WITH_PREFIX_11(C,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) _EXPAND_WITH_PREFIX_START##C pfx a1 , pfx a2 , pfx a3 , pfx a4 , pfx a5 , pfx a6 , pfx a7 , pfx a8 , pfx a9 , pfx a10
#define _EXPAND_WITH_PREFIX_12(...) EXPAND_WITH_PREFIX_ERROR_TOO_MANY_ARGS
#define _EXPAND_WITH_PREFIX_COUNT(...) _EXPAND_WITH_PREFIX_SELECT_13(__VA_ARGS__, 12 , 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 )
#define _EXPAND_WITH_PREFIX_SELECT_13(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 , a12 , a13 , ...) a13


// expand with prefix function macro
#define EXPAND_WITH_FUNC_NOSEP(...)                                  _EXPAND_WITH_FUNC_HELPER1(0,0,_EXPAND_WITH_FUNC_COUNT(__VA_ARGS__),##__VA_ARGS__)
#define EXPAND_WITH_FUNC(...)                                        _EXPAND_WITH_FUNC_HELPER1(0,1,_EXPAND_WITH_FUNC_COUNT(__VA_ARGS__),##__VA_ARGS__)
#define EXPAND_WITH_FUNC_PREPEND_COMMA(...)                          _EXPAND_WITH_FUNC_HELPER1(1,1,_EXPAND_WITH_FUNC_COUNT(__VA_ARGS__),##__VA_ARGS__)
#define _EXPAND_WITH_FUNC_HELPER1(C,S,N,pfx,...)                     _EXPAND_WITH_FUNC_HELPER2(C,S,N,pfx,##__VA_ARGS__)
#define _EXPAND_WITH_FUNC_HELPER2(C,S,N,pfx,...)                     _EXPAND_WITH_FUNC_HELPER3(C,S,N,pfx,##__VA_ARGS__)
#define _EXPAND_WITH_FUNC_HELPER3(C,S,N,pfx,...)                     _EXPAND_WITH_FUNC_##N(C,S,pfx,##__VA_ARGS__)

/*********************** generated code ****************************/
/*
# copy-paste following to generate code
python3 <<EOF
N=15
for i in range(N+1):
    print("#define _EXPAND_WITH_FUNC_%d(C,S,pfx"%(i+1),end='')
    for j in range(i): print(",a%d"%(j+1),end='')
    print(")",end='')
    psep="_EXPAND_WITH_PREFIX_START##C"
    for j in range(i):
        print(" %s pfx(a%d)" % (psep,j+1) , end='' )
        psep="_EXPAND_WITH_SEP##S"
    print("")
print("#define _EXPAND_WITH_FUNC_%d(...) EXPAND_WITH_FUNC_ERROR_TOO_MANY_ARGS"%(N+2))
print("#define _EXPAND_WITH_FUNC_COUNT(...) _EXPAND_WITH_FUNC_SELECT_%d(__VA_ARGS__"%(N+3),end='')
for i in range(N+3): print(",%d"%(N+2-i),end='')
print(")")
print("#define _EXPAND_WITH_FUNC_SELECT_%d("%(N+3),end='')
for i in range(N+3): print("a%d,"%(i+1),end='')
print("...) a%d"%(N+3))
EOF
*/
#define _EXPAND_WITH_FUNC_1(C,S,pfx)
#define _EXPAND_WITH_FUNC_2(C,S,pfx,a1) _EXPAND_WITH_PREFIX_START##C pfx(a1)
#define _EXPAND_WITH_FUNC_3(C,S,pfx,a1,a2) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2)
#define _EXPAND_WITH_FUNC_4(C,S,pfx,a1,a2,a3) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3)
#define _EXPAND_WITH_FUNC_5(C,S,pfx,a1,a2,a3,a4) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4)
#define _EXPAND_WITH_FUNC_6(C,S,pfx,a1,a2,a3,a4,a5) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5)
#define _EXPAND_WITH_FUNC_7(C,S,pfx,a1,a2,a3,a4,a5,a6) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6)
#define _EXPAND_WITH_FUNC_8(C,S,pfx,a1,a2,a3,a4,a5,a6,a7) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7)
#define _EXPAND_WITH_FUNC_9(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8)
#define _EXPAND_WITH_FUNC_10(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9)
#define _EXPAND_WITH_FUNC_11(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10)
#define _EXPAND_WITH_FUNC_12(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10) _EXPAND_WITH_SEP##S pfx(a11)
#define _EXPAND_WITH_FUNC_13(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10) _EXPAND_WITH_SEP##S pfx(a11) _EXPAND_WITH_SEP##S pfx(a12)
#define _EXPAND_WITH_FUNC_14(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10) _EXPAND_WITH_SEP##S pfx(a11) _EXPAND_WITH_SEP##S pfx(a12) _EXPAND_WITH_SEP##S pfx(a13)
#define _EXPAND_WITH_FUNC_15(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10) _EXPAND_WITH_SEP##S pfx(a11) _EXPAND_WITH_SEP##S pfx(a12) _EXPAND_WITH_SEP##S pfx(a13) _EXPAND_WITH_SEP##S pfx(a14)
#define _EXPAND_WITH_FUNC_16(C,S,pfx,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) _EXPAND_WITH_PREFIX_START##C pfx(a1) _EXPAND_WITH_SEP##S pfx(a2) _EXPAND_WITH_SEP##S pfx(a3) _EXPAND_WITH_SEP##S pfx(a4) _EXPAND_WITH_SEP##S pfx(a5) _EXPAND_WITH_SEP##S pfx(a6) _EXPAND_WITH_SEP##S pfx(a7) _EXPAND_WITH_SEP##S pfx(a8) _EXPAND_WITH_SEP##S pfx(a9) _EXPAND_WITH_SEP##S pfx(a10) _EXPAND_WITH_SEP##S pfx(a11) _EXPAND_WITH_SEP##S pfx(a12) _EXPAND_WITH_SEP##S pfx(a13) _EXPAND_WITH_SEP##S pfx(a14) _EXPAND_WITH_SEP##S pfx(a15)
#define _EXPAND_WITH_FUNC_17(...) EXPAND_WITH_FUNC_ERROR_TOO_MANY_ARGS
#define _EXPAND_WITH_FUNC_COUNT(...) _EXPAND_WITH_FUNC_SELECT_18(__VA_ARGS__,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define _EXPAND_WITH_FUNC_SELECT_18(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,...) a18
/********************************* end of generated code *******************************************/


#define GET_FIRST_ARG(XXX,...) XXX

#define _EXPAND_WITH_PREFIX_START0 /**/
#define _EXPAND_WITH_PREFIX_START1 ,

#define _EXPAND_WITH_SEP0 /**/
#define _EXPAND_WITH_SEP1 ,
/*********************************************************************************************************************/



#if defined(__INTEL_COMPILER) || defined(__clang__)
  #define OPT_COMMA_VA_ARGS(...) ,##__VA_ARGS__
#else
  #define OPT_COMMA_VA_ARGS(...) __VA_OPT__(,)__VA_ARGS__
#endif

// utility macros
#define ONIKA_CONCAT(a,b) _ONIKA_CONCAT(a,b)
#define _ONIKA_CONCAT(a,b) a##b

#define ONIKA_STR(x) _ONIKA_STR(x)
#define _ONIKA_STR(x) #x


// capture workaround, creates real C++ variables from binding references, to avoid llvm errors when capturing these references in other lambdas
#define _onika_bind_vars_pfxtmp(x) __##x
#define _onika_bind_vars_copy(x) auto x=__##x;
#define _onika_bind_vars_move(x) auto x=std::move(__##x);

#define onika_bind_vars_copy(s,...) \
  auto [ EXPAND_WITH_FUNC(_onika_bind_vars_pfxtmp OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ] = s; \
  EXPAND_WITH_FUNC_NOSEP(_onika_bind_vars_copy OPT_COMMA_VA_ARGS(__VA_ARGS__) )

#define onika_bind_vars_move(s,...) \
  auto [ EXPAND_WITH_FUNC(_onika_bind_vars_pfxtmp OPT_COMMA_VA_ARGS(__VA_ARGS__) ) ] = s; \
  EXPAND_WITH_FUNC_NOSEP(_onika_bind_vars_move OPT_COMMA_VA_ARGS(__VA_ARGS__) )

#define onika_bind_vars(...) onika_bind_vars_copy(__VA_ARGS__)

// utilities to reference unsused variables
#define FAKE_USE_VAR(x) if(sizeof(decltype(x))==0){}
#define FAKE_USE_OF_VARIABLES(...) EXPAND_WITH_FUNC_NOSEP( FAKE_USE_VAR OPT_COMMA_VA_ARGS(__VA_ARGS__) )

