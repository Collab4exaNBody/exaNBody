#pragma once

#include <cstdint>
#include <onika/oarray.h>

namespace onika
{

  namespace dac
  {

    template<size_t Nd> struct ItemCoordTypeHelper { using item_coord_t = oarray_t<size_t,Nd>; static inline constexpr item_coord_t zero = {}; };
    template<> struct ItemCoordTypeHelper<1> { using item_coord_t = oarray_t<size_t,1>; static inline constexpr item_coord_t zero = {0}; };
    template<> struct ItemCoordTypeHelper<2> { using item_coord_t = oarray_t<size_t,2>; static inline constexpr item_coord_t zero = {0,0}; };
    template<> struct ItemCoordTypeHelper<3> { using item_coord_t = oarray_t<size_t,3>; static inline constexpr item_coord_t zero = {0,0,0}; };
    template<> struct ItemCoordTypeHelper<4> { using item_coord_t = oarray_t<size_t,4>; static inline constexpr item_coord_t zero = {0,0,0,0}; };

    template<size_t Nd> using item_nd_coord_t = typename ItemCoordTypeHelper<Nd>::item_coord_t;

  }
  
}


