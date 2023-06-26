#pragma once

#include <cstdint>
#include <onika/oarray.h>
#include <onika/cuda/cuda.h>
#include <optional>

#include <optional>

namespace onika
{

  namespace dac
  {
    template<class T>
    struct Array1DView
    {
      //struct _unused_type{};
      T* const m_start = nullptr;
      const size_t m_size = 0;
      inline auto data() const { return m_start; }
      inline auto size() const { return m_size; }
      inline T& operator [] (size_t i) const { return m_start[i]; }
      inline operator std::conditional_t< std::is_const_v<T> , std::nullopt_t , Array1DView<const T> > () const
      {
          if constexpr ( ! std::is_const_v<T> ) return { m_start , m_size };
          return {};
      }
    };

    template<class T>
    struct Array3DView
    {
      T* const m_start = nullptr;
      oarray_t<size_t,3> m_size = { 0 , 0 , 0 };
      ONIKA_HOST_DEVICE_FUNC inline auto data() const { return m_start; }
      ONIKA_HOST_DEVICE_FUNC inline auto size() const { return m_size; }
    };

    template<class T>
    struct MultiValueArray3DView
    {
      T* const m_start = nullptr;
      size_t m_components = 0; // number of values
      oarray_t<size_t,3> m_size = { 0 , 0 , 0 };
      inline auto data() const { return m_start; }
      inline auto size() const { return m_size; }
      inline size_t components() const { return m_components; }
    };

    template<class T>
    static inline Array3DView<T> make_array_3d_view( T* data , const oarray_t<size_t,3>& sz )
    {
      return { data , sz };
    }

/*    template<class T, class A>
    static inline Array3DView<T> make_array_3d_view( std::vector<T,A>& vec , const oarray_t<size_t,3>& sz )
    {
      return { vec.data() , sz };
    }
*/

    template<class T>
    static inline MultiValueArray3DView<T> make_nvalues_array_3d_view( T* data , size_t ncomps, const oarray_t<size_t,3>& sz )
    {
      return { data , ncomps , sz };
    }


  }

}

