#pragma once

#include <atomic>

namespace onika
{

# ifdef __HIPCC__
# define _XNB_ATOMIC_FUNC_DECL __host__ __device__
# else
# define _XNB_ATOMIC_FUNC_DECL
# endif

  template<class T>
  static inline _XNB_ATOMIC_FUNC_DECL T capture_atomic_add(T& x , const T& a)
  {
    T r;
#   pragma omp atomic capture
    { r = x ; x += a; }
    return r;
  }

  template<class T>
  static inline _XNB_ATOMIC_FUNC_DECL T capture_atomic_sub(T& x , const T& a)
  {
    T r;
#   pragma omp atomic capture
    { r = x ; x -= a; }
    return r;
  }

  template<class T>
  static inline _XNB_ATOMIC_FUNC_DECL void inplace_atomic_add(T& x , const T& a)
  {
#   pragma omp atomic update
    x += a;
  }

  template<class T>
  static inline _XNB_ATOMIC_FUNC_DECL void inplace_atomic_sub(T& x , const T& a)
  {
#   pragma omp atomic update
    x -= a;
  }

  template<class T>
  static inline T capture_atomic_min(T& x , const T& a)
  {
    auto * atomic_ptr = reinterpret_cast< std::atomic<std::remove_cv_t<std::remove_reference_t<decltype(x)> > > * >( &x );
    T r, xval = atomic_ptr->load();
    do { r = ( xval < a ) ? xval : a; } while( ! atomic_ptr->compare_exchange_weak(xval,r) );
    return xval;
  }

  template<class T>
  static inline T capture_atomic_max(T& x , const T& a)
  {
    auto * atomic_ptr = reinterpret_cast< std::atomic<std::remove_cv_t<std::remove_reference_t<decltype(x)> > > * >( &x );
    T r, xval = atomic_ptr->load();
    do { r = ( xval > a ) ? xval : a; } while( ! atomic_ptr->compare_exchange_weak(xval,r) );
    return xval;
  }

}

