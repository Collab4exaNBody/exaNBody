#pragma once

#include <cstdint>
#include <onika/oarray.h>
#include <vector>

#include <onika/dac/item_coord.h>
#include <onika/dac/array_view.h>
#include <onika/dac/slices.h>

#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>

namespace onika
{

  namespace dac
  {

    template<typename T>
    struct DataDecompositionTraits
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static inline constexpr size_t ND = 0;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = T;
      using pointer_t = value_t*;
      using reference_t = T&;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) &v; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr item_coord_t size(reference_t) { return {}; }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t count(reference_t) { return 1; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(reference_t v , item_coord_t) { return v; }
    };

    template<typename T, typename A>
    struct DataDecompositionTraits< std::vector<T,A> >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 1;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = std::vector<T,A>;
      using pointer_t = value_t*;
      using reference_t = value_t &;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) v.data(); }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(reference_t v, const item_coord_t& c) { return (void*)( & v[c.x[0]] ); }
      static inline item_coord_t size(reference_t v) { return { v.size() }; }
      static inline size_t count(reference_t v) { return v.size(); }
      static inline item_t& at(reference_t v , const item_coord_t& c) { return v[c.x[0]]; }
    };

    template<typename T>
    struct DataDecompositionTraits< Array1DView<T> >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 1;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using reference_t = Array1DView<T>;
      using pointer_t = T*;
      using value_t = T[];
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(const reference_t& v) { return (void*) v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(const reference_t& v) { return v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(const reference_t& v, const item_coord_t& c) { return (void*)( v.m_start + c.x[0] ); }
      ONIKA_HOST_DEVICE_FUNC static inline item_coord_t size(const reference_t& v) { return { v.m_size }; }
      ONIKA_HOST_DEVICE_FUNC static inline size_t count(const reference_t& v) { return v.m_size; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at( const reference_t& v , const item_coord_t& c) { assert(c.x[0]<v.m_size); return v.m_start[c.x[0]]; }
    };

    template<typename T>
    struct DataDecompositionTraits< Array3DView<T> >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 3;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using reference_t = Array3DView<T>;
      using pointer_t = T*;
      using value_t = T[];
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(const reference_t& v) { return v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(const reference_t& v, const item_coord_t& c) { return (void*)( v.m_start + coord_to_index(c,v.m_size) ); }
      ONIKA_HOST_DEVICE_FUNC static inline item_coord_t size(const reference_t& v) { return v.m_size; }
      ONIKA_HOST_DEVICE_FUNC static inline size_t count(const reference_t& v) { return v.m_size[0] * v.m_size[1] * v.m_size[2]; }      
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(const reference_t& v , const item_coord_t& c) { return v.m_start[coord_to_index(c,v.m_size)]; }
    };

    template<typename T>
    struct DataDecompositionTraits< MultiValueArray3DView<T> >
    {
      using slices_t = typename DataSlicing< Array1DView<T> >::slices_t;
      static constexpr size_t ND = 3;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using reference_t = MultiValueArray3DView<T>;
      using pointer_t = T*;
      using value_t = T[];
      using item_t = Array1DView<T>;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(const reference_t& v) { return v.m_start; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(const reference_t& v, const item_coord_t& c) { return (void*)( v.m_start + coord_to_index(c,v.m_size) * v.m_components ); }
      ONIKA_HOST_DEVICE_FUNC static inline item_coord_t size(const reference_t& v) { return v.m_size; }
      ONIKA_HOST_DEVICE_FUNC static inline size_t count(const reference_t& v) { return v.m_size[0] * v.m_size[1] * v.m_size[2]; }      
      ONIKA_HOST_DEVICE_FUNC static inline item_t at(const reference_t& v , const item_coord_t& c) { return { v.m_start + coord_to_index(c,v.m_size) * v.m_components , v.m_components }; }
    };

    template<typename T, size_t N>
    struct DataDecompositionTraits< T[N] >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 1;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = T[N];
      using reference_t = value_t &;
      using pointer_t = value_t *;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) &v; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(reference_t v, const item_coord_t& c) { return (void*)( & v[c.x[0]] ); }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr item_coord_t size(reference_t) { return { N }; }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t count(reference_t) { return N; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(reference_t v , const item_coord_t& c) { return v[c.x[0]]; }
    };

    template<typename T, size_t W, size_t H>
    struct DataDecompositionTraits< T[H][W] >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 2;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = T[H][W];
      using reference_t = value_t &;
      using pointer_t = value_t *;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) &v; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(reference_t v, const item_coord_t& c) { return (void*)( & v[c.x[1]][c.x[0]] ); }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr item_coord_t size(reference_t) { return { W , H }; }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t count(reference_t) { return W * H; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(reference_t v , const item_coord_t& c) { return v[c.x[1]][c.x[0]]; }
    };

    template<typename T, size_t W, size_t H, size_t Z>
    struct DataDecompositionTraits< T[Z][H][W] >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 3;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = T[Z][H][W];
      using reference_t = value_t &;
      using pointer_t = value_t *;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline void* identifier(reference_t v) { return (void*) &v; }
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(reference_t v, const item_coord_t& c) { return (void*)( & v[c.x[2]][c.x[1]][c.x[0]] ); }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr item_coord_t size(reference_t) { return { W , H , Z }; }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t count(reference_t) { return W * H * Z; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(reference_t v , const item_coord_t& c) { return v[c[2]][c[1]][c[0]]; }
    };
   
  }
}

