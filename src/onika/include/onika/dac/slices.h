#pragma once

#include <onika/dac/constants.h>
#include <onika/flat_tuple.h>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <cassert>

#include <onika/cuda/cuda.h>
#include <onika/dac/array_view.h>

#define __DAC_AUTO_RET_EXPR( ... ) -> decltype( __VA_ARGS__ ) { return __VA_ARGS__ ; }


#include <iostream>

namespace onika
{

  namespace dac
  {

    // utility template, that may be specialized to provide user friendly slice name
    template<class T> struct DataSliceName
    {
      static inline const char* cstr() { return typeid(T).name(); }
    };
    template<class T> static inline const char* data_slice_name_v = DataSliceName<T>::cstr();

    // ============= data slices ===============
    namespace details
    {
      template<typename U, typename... T> struct FindArgPlace;
      template<typename U> struct FindArgPlace<U> { static constexpr int place = std::numeric_limits<int>::lowest(); };
      template<typename U, typename T0, typename... T> struct FindArgPlace<U,T0,T...> { static constexpr int place = std::is_same_v<U,T0> ? 0 : FindArgPlace<U,T...>::place+1; };
      template<typename... T> struct FirstTemplateArg { using type = void; };
      template<typename T0, typename... T> struct FirstTemplateArg<T0,T...> { using type = T0; };
      template<class... T> using first_template_arg_t = typename FirstTemplateArg<T...>::type;
    }

    template<class T , class A> struct DataSliceAccess
    {
      using slice_t = T;
      using access_mode_t = A;
    };
        
    template<typename... T>
    struct DataSlices
    {
      static inline constexpr size_t nb_slices = sizeof...(T);
      static inline constexpr uint64_t all_slices_mask = (1ul<<nb_slices) - 1;
      using first_slice_t = details::first_template_arg_t<T...>;
      template<typename U> static inline constexpr ssize_t slice_index_weak( U )
      {
        return details::FindArgPlace<U,T...>::place;
      }
      template<typename U> static inline constexpr size_t slice_index( U )
      {
        constexpr ssize_t i = details::FindArgPlace<U,T...>::place;
        static_assert( i>=0 && i<nb_slices , "slice not found" );
        return i;
      }
      template<typename U> static inline constexpr uint64_t bit_mask( U )
      {
        constexpr ssize_t i = details::FindArgPlace<U,T...>::place;
        static_assert( i>=0 && i<nb_slices , "slice not found" );
        return 1ull << i;
      }
    };

    template<typename... T> struct DataSliceAccesses
    {
      using slices_t = DataSlices< typename T::slice_t ... >;
      static inline constexpr bool all_ro = ( ... && ( std::is_same_v<typename T::access_mode_t,dac::ro_t> ) );
      static inline constexpr size_t nb_slices = sizeof...(T);
      static inline constexpr size_t nb_ro_slices = ( ... + ( size_t(std::is_same_v<typename T::access_mode_t,dac::ro_t>) ) );
      static inline constexpr size_t nb_rw_slices = ( ... + ( size_t(std::is_same_v<typename T::access_mode_t,dac::rw_t>) ) );
      template<class U> struct Concat;
      template<class... U> struct Concat< DataSliceAccesses<U...> > { using type = DataSliceAccesses< T... , U ... >; };
    };
    
    template<class _AccessMode , class _Slices> struct DataSlicesToDataSliceAccesses;
    template<class _AccessMode , class... T> struct DataSlicesToDataSliceAccesses<_AccessMode , DataSlices<T...> > { using type = DataSliceAccesses< DataSliceAccess<T,_AccessMode> ... >; };
    template<class A, class S> using slices_to_slice_accesses_t = typename DataSlicesToDataSliceAccesses<A,S>::type;
    template<class RoSAcc , class RwSAcc> using concat_slice_accesses_t = typename RoSAcc:: template Concat<RwSAcc>::type;

    // default slicing for an unknown scalar type : one single slice representing the whole data
    template<typename T>
    struct DataSlicing
    {
      using slices_t = DataSlices< whole_t >;
      using value_t = T;
      using reference_t = value_t &;
      using const_reference_t = const value_t &;
      ONIKA_HOST_DEVICE_FUNC static inline const_reference_t get_slice(const_reference_t v , DataSliceAccess<whole_t,ro_t> ) { return v;  }
      ONIKA_HOST_DEVICE_FUNC static inline       reference_t get_slice(      reference_t v , DataSliceAccess<whole_t,rw_t> ) { return v; }
    };

    template<typename T>
    struct DataSlicing< Array1DView<T> >
    {
      using slices_t = DataSlices< whole_t >;
      using value_t = Array1DView<T>;
      using reference_t = Array1DView<T>;
      using const_reference_t = Array1DView<const T>;
      ONIKA_HOST_DEVICE_FUNC static inline const_reference_t get_slice(const value_t& v , DataSliceAccess<whole_t,ro_t> ) { return { v.m_start , v.m_components , v.m_size };  }
      ONIKA_HOST_DEVICE_FUNC static inline       reference_t get_slice(const value_t& v , DataSliceAccess<whole_t,rw_t> ) { return v; }
    };

    // sample slicing definition for a pair. two slices, first and second
    template<class A, class B>
    struct DataSlicing< std::pair<A,B> >
    {
      using slices_t = DataSlices< pair_first_t , pair_second_t >;
      using value_t = std::pair<A,B>;
      using reference_t = value_t &;
      using const_reference_t = const value_t &;
      ONIKA_HOST_DEVICE_FUNC static inline const auto& get_slice(const value_t& v , DataSliceAccess<pair_first_t,ro_t> ) { return v.first; }
      ONIKA_HOST_DEVICE_FUNC static inline const auto& get_slice(const value_t& v , DataSliceAccess<pair_second_t,ro_t> ) { return v.second; }
      ONIKA_HOST_DEVICE_FUNC static inline auto& get_slice(value_t& v , DataSliceAccess<pair_first_t,rw_t> ) { return v.first; }
      ONIKA_HOST_DEVICE_FUNC static inline auto& get_slice(value_t& v , DataSliceAccess<pair_second_t,rw_t> ) { return v.second; }
    };

    // helper to compute the bitmask of a subset of a set of slices
    template<typename DS, typename DSSub> struct DataSliceSubSetMask ;
    template<typename DS> struct DataSliceSubSetMask<DS,DataSlices<> > { static inline constexpr uint64_t value = 0; };
    template<typename DS, typename... S> struct DataSliceSubSetMask<DS,DataSlices<S...> > { static inline constexpr uint64_t value = ( ... | ( DS::bit_mask(S{}) ) ); };
    template<typename DS, typename DSSub> static inline constexpr uint64_t subset_bit_mask_v = DataSliceSubSetMask<DS,DSSub>::value;

    // ===== access to a subset of data slices =====
    template<  typename T
             , typename DSSub
             , bool = (DSSub::nb_slices==1) 
             , bool = ( DataSlicing<T>::slices_t::all_slices_mask == subset_bit_mask_v<typename DataSlicing<T>::slices_t,typename DSSub::slices_t> )
             , bool = DSSub::all_ro >
    struct DataSlicesSubSet ;

    // general case, return a tuple of slice accessors/references
    template<typename T, bool RO, typename... S >
    struct DataSlicesSubSet< T , DataSliceAccesses<S...> , false , false , RO >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      using subset_t = typename DataSliceAccesses<S...>::slices_t;
      using subset_access_t = DataSliceAccesses<S...>;
      using value_t = typename DataSlicing<T>::value_t;
      static inline constexpr uint64_t bit_mask_v = subset_bit_mask_v<slices_t,subset_t>;
      ONIKA_HOST_DEVICE_FUNC static inline auto get_slices(value_t& v) __DAC_AUTO_RET_EXPR( make_flat_tuple( DataSlicing<T>::get_slice(v,S{}) ... ) )
    };

    // specialization when the subset is the whole set
    template<typename T, bool ONE, typename... S >
    struct DataSlicesSubSet< T , DataSliceAccesses<S...> , ONE , true , true >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      using subset_t = typename DataSliceAccesses<S...>::slices_t;
      using subset_access_t = DataSliceAccesses<S...>;
      using value_t = typename DataSlicing<T>::value_t;
      using reference_t = typename DataSlicing<T>::reference_t;
      using const_reference_t = typename DataSlicing<T>::const_reference_t;
      static inline constexpr uint64_t bit_mask_v = subset_bit_mask_v<slices_t,subset_t>;
      ONIKA_HOST_DEVICE_FUNC static inline const_reference_t get_slices(reference_t v) { return v; }
    };
    template<typename T, bool ONE, typename... S >
    struct DataSlicesSubSet< T , DataSliceAccesses<S...> , ONE , true , false >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      using subset_t = typename DataSliceAccesses<S...>::slices_t;
      using subset_access_t = DataSliceAccesses<S...>;
      using value_t = typename DataSlicing<T>::value_t;
      using reference_t = typename DataSlicing<T>::reference_t;
      static inline constexpr uint64_t bit_mask_v = subset_bit_mask_v<slices_t,subset_t>;
      ONIKA_HOST_DEVICE_FUNC static inline reference_t get_slices(reference_t v) { return v; }
    };

    // specialization when the subet contains only one element among several
    template<typename T, bool RO , typename S >
    struct DataSlicesSubSet< T , DataSliceAccesses<S> , true , false , RO >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      using subset_t = typename DataSliceAccesses<S>::slices_t;
      using subset_access_t = DataSliceAccesses<S>;
      using value_t = typename DataSlicing<T>::value_t;
      static inline constexpr uint64_t bit_mask_v = subset_bit_mask_v<slices_t,subset_t>;
      ONIKA_HOST_DEVICE_FUNC static inline auto get_slices(value_t& v) __DAC_AUTO_RET_EXPR( DataSlicing<T>::get_slice(v,S{}) )
    };
  
    // =======================================
    
        
    // =========== data slice to pointer mapping ==========
    template<unsigned int s> struct slice_size_po2 { static constexpr unsigned int value = 8; };
    template<> struct slice_size_po2<1> { static constexpr unsigned int value = 1; };
    template<> struct slice_size_po2<2> { static constexpr unsigned int value = 2; };
    template<> struct slice_size_po2<3> { static constexpr unsigned int value = 2; };
    template<> struct slice_size_po2<4> { static constexpr unsigned int value = 4; };
    template<> struct slice_size_po2<5> { static constexpr unsigned int value = 4; };
    template<> struct slice_size_po2<6> { static constexpr unsigned int value = 4; };
    template<> struct slice_size_po2<7> { static constexpr unsigned int value = 4; };

    template<typename T>
    struct DataSliceAddress
    {
      using slices_t = typename DataSlicing<T>::slices_t;    
      static constexpr size_t nb_slices = DataSlicing<T>::slices_t::nb_slices;
      static constexpr size_t obj_size = sizeof(T);
      static_assert( (obj_size / nb_slices) > 0 , "Impossible automatic conversion of slice to pointer" );
      static constexpr size_t slice_size = slice_size_po2< obj_size / nb_slices >::value;
      static inline void* slice_address( T* obj_ptr , unsigned int i )
      {
        assert( /* i>=0 && */ i<nb_slices );
        return ((uint8_t*)obj_ptr) + i * slice_size;
      }
    };
    
  }
}

#undef __DAC_AUTO_RET_EXPR


