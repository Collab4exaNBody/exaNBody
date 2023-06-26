#pragma once

#include <type_traits>

#define ONIKA_SILENT_USE(x) if(((&(x))-(std::remove_reference_t<decltype(x)>*)nullptr)==0){}

