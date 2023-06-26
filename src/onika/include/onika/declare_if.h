#pragma once

#include <type_traits>

#define DECLARE_IF_CONSTEXPR(c,t,v) std::conditional_t<c,t,onika::BoolConst<false> > v = {}; if constexpr (!c) if(v) (void)0

