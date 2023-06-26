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

