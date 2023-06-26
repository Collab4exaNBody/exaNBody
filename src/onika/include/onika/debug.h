#pragma once

#ifndef NDEBUG
#define ONIKA_DEBUG_ONLY(...) __VA_ARGS__
#else
#define ONIKA_DEBUG_ONLY(...) /**/
#endif


