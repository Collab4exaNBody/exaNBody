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

// utility macros
#define USTAMP_CONCAT(a,b) _USTAMP_CONCAT(a,b)
#define _USTAMP_CONCAT(a,b) a##b

#define USTAMP_STR(x) _USTAMP_STR(x)
#define _USTAMP_STR(x) #x

#ifndef XSTAMP_TARGET_ID
#define XSTAMP_TARGET_ID undefined_target
#endif

#ifndef XSTAMP_SOURCE_ID
#define XSTAMP_SOURCE_ID undefined_source
#endif

#define MAKE_UNIQUE_NAME(base,sep,I,J) _MAKE_UNIQUE_NAME(base,sep,I,J)
#define _MAKE_UNIQUE_NAME(base,sep,I,J) base##sep##I##sep##J

#define CONSTRUCTOR_ATTRIB __attribute__((constructor))
#define CONSTRUCTOR_FUNCTION CONSTRUCTOR_ATTRIB void MAKE_UNIQUE_NAME(__exanb_constructor,_,XSTAMP_TARGET_ID,XSTAMP_SOURCE_ID) ()
#define CONSTRUCTOR_FUNCTION_VARIANT(name) CONSTRUCTOR_ATTRIB void MAKE_UNIQUE_NAME(__exanb_constructor_##name,_,XSTAMP_TARGET_ID,XSTAMP_SOURCE_ID) ()

#include <onika/macro_utils.h>

