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

#include <cstdint>
#include <onika/oarray.h>
#include <vector>

#include <onika/dac/decomposition.h>
#include <onika/dac/stencil.h>
#include <onika/dac/reduction.h>

#include <onika/cuda/cuda.h>

#define __DAC_AUTO_RET_EXPR( ... ) -> decltype( __VA_ARGS__ ) { return __VA_ARGS__ ; }

namespace onika
{

  namespace dac
  {

    template<size_t N>
    struct FakeDataAccessControler
    {
      static constexpr size_t ND = N;
      template<size_t M> ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at( oarray_t<size_t,M> = oarray_t<size_t,M>{} ) { return nullptr; }
    };

  
    template<class T, class AccessStencilT, class DDT = DataDecompositionTraits<T> , bool Is0D = (DDT::ND==0) , class ReduceOp = std::nullptr_t /* do not use when no explicit type */ >
    struct DataAccessControler
    {
      static inline constexpr size_t ND = DDT::ND;
      using ddt_t = DDT;
      using item_coord_t = typename DDT::item_coord_t;
      static inline constexpr item_coord_t zero_coord = DDT::zero_coord;
      using value_t = typename DDT::value_t;
      using reference_t = typename DDT::reference_t;
      using pointer_t = typename DDT::pointer_t;
      using item_t = typename DDT::item_t;
      using slices_t = typename DDT::slices_t;
      static inline constexpr size_t nb_slices = slices_t::nb_slices;
      using access_stencil_t = AccessStencilT;
      using slices_subset_t = DataSlicesSubSet< item_t , typename access_stencil_t::central_t::ro_rw_slices_t >;
      template <size_t Ne> using multiple_coord_t = oarray_t<item_coord_t,Ne>;
      static_assert( std::is_same_v< slices_t , typename slices_subset_t::slices_t > , "inconsistent slice set between DDT and SubSet" );      
      ONIKA_HOST_DEVICE_FUNC inline size_t count() const { return ddt_t::count( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline item_coord_t size() const { return ddt_t::size( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline pointer_t pointer() const { return ddt_t::pointer( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline auto pointer_at( item_coord_t c ) const -> decltype(ddt_t::pointer_at(std::declval<reference_t>(),c))
      {
        auto p = ddt_t::pointer_at(m_ref,c);
        assert(p != nullptr);
        return p;
      }
      ONIKA_HOST_DEVICE_FUNC inline auto at( item_coord_t c ) const -> decltype( slices_subset_t::get_slices( ddt_t::at(std::declval<reference_t>(),c) ) )
      {
        return slices_subset_t::get_slices( ddt_t::at(m_ref,c) );
      }
      reference_t m_ref;
    };

    template<class T, class AccessStencilT, class DDT>
    struct DataAccessControler<T,AccessStencilT,DDT,true,std::nullptr_t>
    {
      static inline constexpr size_t ND = 0;
      using ddt_t = DDT;
      using item_coord_t = typename DDT::item_coord_t;
      static inline constexpr item_coord_t zero_coord = DDT::zero_coord;
      using value_t = typename DDT::value_t;
      using reference_t = typename DDT::reference_t;
      using pointer_t = typename DDT::pointer_t;
      using item_t = std::conditional_t< DDT::ND == 0 , typename DDT::item_t , value_t >;
      using slices_t = typename DDT::slices_t;
      static inline constexpr size_t nb_slices = slices_t::nb_slices;
      using access_stencil_t = AccessStencilT;
      using slices_subset_t = DataSlicesSubSet< item_t , typename access_stencil_t::central_t::ro_rw_slices_t >;
      template <size_t Ne> using multiple_coord_t = oarray_t<item_coord_t,Ne>;
      static_assert( std::is_same_v< slices_t , typename slices_subset_t::slices_t > , "inconsistent slice set between DDT and SubSet" );      
      ONIKA_HOST_DEVICE_FUNC inline size_t count() const { return ddt_t::count( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline item_coord_t size() const { return ddt_t::size( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline pointer_t pointer() const { return ddt_t::pointer( m_ref ); }
      template<size_t M=0>
      ONIKA_HOST_DEVICE_FUNC inline auto pointer_at( oarray_t<size_t,M> = oarray_t<size_t,M>{} ) const -> decltype(ddt_t::identifier(std::declval<reference_t>()))
      {
        return ddt_t::identifier(m_ref);
      }
      template<size_t M=0>
      ONIKA_HOST_DEVICE_FUNC inline auto at( oarray_t<size_t,M> = oarray_t<size_t,M>{} ) const -> decltype(slices_subset_t::get_slices(std::declval<reference_t>()))
      {
        static_assert( M==0 || access_stencil_t::central_t::rw_slices_t::nb_slices == 0 , "auxiliar scalar accessor must be entirely read-only");
        return slices_subset_t::get_slices( m_ref );
      }
      reference_t m_ref;
    };

    template<class T, class AccessStencilT, class DDT, class ReduceOp>
    struct DataAccessControler<T,AccessStencilT,DDT,true,ReduceOp>
    {
      static_assert( ! std::is_same_v<ReduceOp,std::nullptr_t> , "template specialization error" );
      static inline constexpr size_t ND = 0;
      using ddt_t = DDT;
      using item_coord_t = typename DDT::item_coord_t;
      static inline constexpr item_coord_t zero_coord = DDT::zero_coord;
      using value_t = typename DDT::value_t;
      using reference_t = typename DDT::reference_t;
      using pointer_t = typename DDT::pointer_t;
      using item_t = std::conditional_t< DDT::ND == 0 , typename DDT::item_t , value_t >;
      using slices_t = typename DDT::slices_t;
      static inline constexpr size_t nb_slices = slices_t::nb_slices;
      using access_stencil_t = AccessStencilT;
      using slices_subset_t = DataSlicesSubSet< item_t , typename access_stencil_t::central_t::ro_rw_slices_t >;
      template <size_t Ne> using multiple_coord_t = oarray_t<item_coord_t,Ne>;
      static_assert( std::is_same_v< slices_t , typename slices_subset_t::slices_t > , "inconsistent slice set between DDT and SubSet" );      
      ONIKA_HOST_DEVICE_FUNC inline size_t count() const { return ddt_t::count( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline item_coord_t size() const { return ddt_t::size( m_ref ); }
      ONIKA_HOST_DEVICE_FUNC inline pointer_t pointer() const { return ddt_t::pointer( m_ref ); }
      template<size_t M=0>
      ONIKA_HOST_DEVICE_FUNC inline auto pointer_at( oarray_t<size_t,M> = oarray_t<size_t,M>{} ) const -> decltype(ddt_t::identifier(std::declval<reference_t>()))
      {
        return ddt_t::identifier(m_ref);
      }
      template<size_t M=0>
      ONIKA_HOST_DEVICE_FUNC inline reduction_wrapper_t<item_t,ReduceOp> at( oarray_t<size_t,M> = oarray_t<size_t,M>{} ) const
      {
        static_assert( access_stencil_t::central_t::ro_slices_t::nb_slices == 0 , "reduction scalar accessor must be entirely read-write");
//        reduction_wrapper_t<item_t,ReduceOp> r { m_ref };
        return { m_ref };
      }
      reference_t m_ref;
    };


    // ****************** identify DataAccessControler instanciations *********************
    template<class T> struct IsDataAccessControler : public std::false_type {};
    template<class A, class B, class C, bool D, class E> struct IsDataAccessControler< DataAccessControler<A,B,C,D,E> > : public std::true_type {};
    template<class T> static inline constexpr bool is_data_access_controler_v = IsDataAccessControler< std::decay_t<T> >::value;


    // ************** default stencil definitons *****************
    template<class T> using default_ro_stencil_t = Stencil< stencil_element_t< typename DataDecompositionTraits<T>::slices_t , DataSlices<> > , stencil_elements_t<> >;
    template<class T> static inline constexpr default_ro_stencil_t<T> default_ro_stencil = {};
    template<class T> static inline constexpr default_ro_stencil_t<T> make_default_ro_stencil(const T&) { return default_ro_stencil<T>; }
    
    template<class T> using default_rw_stencil_t = Stencil< stencil_element_t< DataSlices<> , typename DataDecompositionTraits<T>::slices_t > , stencil_elements_t<> >;
    template<class T> static inline constexpr default_rw_stencil_t<T> default_rw_stencil = {};
    template<class T> static inline constexpr default_rw_stencil_t<T> make_default_rw_stencil(T&) { return default_rw_stencil<T>; }


    // ****************** accessor assembly methods ******************
    template< typename T , typename AccessStencilT = Stencil< /*stencil_center_t*/ stencil_element_t< DataSlices<> , typename DataDecompositionTraits<T>::slices_t /*, DataDecompositionTraits<T>::ND*/ > , stencil_elements_t<> > >
    static inline
    onika::dac::DataAccessControler<T,AccessStencilT>
    make_access_controler( T& ref , AccessStencilT = AccessStencilT{} )
    {
      return { ref };
    }

    template< typename T , typename AccessStencilT = Stencil< stencil_element_t< typename DataDecompositionTraits<T>::slices_t , DataSlices<> > , stencil_elements_t<> > >
    static inline
    onika::dac::DataAccessControler< T, AccessStencilT, DataDecompositionTraits<T> , true >
    make_scalar_access_controler( T& ref , AccessStencilT = AccessStencilT{} )
    {
      return { ref };
    }

    template< typename T , class ReduceOp = reduction_add_t >
    static inline
    onika::dac::DataAccessControler< T, Stencil< stencil_element_t< DataSlices<> , typename DataDecompositionTraits<T>::slices_t > , stencil_elements_t<> > , DataDecompositionTraits<T> , true , ReduceOp >
    make_reduction_access_controler( T& ref , ReduceOp = ReduceOp{} )
    {
      return { ref };
    }

    // ******************* automatic DataAccessControler creation from var ****************
    template<class T , bool ReadOnly, bool = is_data_access_controler_v<T> > struct AutoDataAccessControler
    {
      using type = T;
      static inline constexpr const T& convert_scalar(const T& x) { return x; }
    };
    template<class T> struct AutoDataAccessControler<T,true,false>
    {
      static inline auto convert_scalar(const T& x) { return make_scalar_access_controler(x,make_default_ro_stencil(x)); }
      using type = decltype( convert_scalar(std::declval<const T&>()) );
    };
    template<class T> struct AutoDataAccessControler<T,false,false>
    {
      static inline auto convert_scalar(T& x) { return make_scalar_access_controler(x,make_default_rw_stencil(x)); }
      using type = decltype( convert_scalar(std::declval<T&>()) );
    };
    template<bool ReadOnly, class Accessor> using auto_dac_convert_t = typename AutoDataAccessControler<Accessor,ReadOnly>::type;
    template<bool RO, class Accessor> using auto_conv_dac_subset_t = decltype( std::declval<auto_dac_convert_t<RO,Accessor> >().at(typename auto_dac_convert_t<RO,Accessor>::item_coord_t{}) );

    template<bool ReadOnly, class _Accessor>
    static inline auto auto_dac_convert( std::integral_constant<bool,ReadOnly> , _Accessor& acc_or_ref )
    {
      return AutoDataAccessControler<_Accessor,ReadOnly>::convert_scalar(acc_or_ref) ;
    }

    template<bool ReadOnly, class... _Accessor>
    static inline auto auto_dac_tuple_convert( std::integral_constant<bool,ReadOnly> , _Accessor& ... acc_or_ref )
    {
      return make_flat_tuple( AutoDataAccessControler<_Accessor,ReadOnly>::convert_scalar(acc_or_ref) ... );
    }
    template<bool ReadOnly, class AccessorTuple> struct AutoDacScalarConvert;
    template<bool ReadOnly, class ... _Accessor> struct AutoDacScalarConvert< ReadOnly, FlatTuple<_Accessor...> >
    {
      using type = FlatTuple< typename AutoDataAccessControler<_Accessor,ReadOnly>::type ... >;
    };
    template<bool ReadOnly, class AccessorTuple> using auto_dac_tuple_convert_t = typename AutoDacScalarConvert<ReadOnly,AccessorTuple>::type;

    // ********************* set of data accessors, filtering, subsetting *****************
    template<class... Dacs> struct DataAccessControlerSet ;
    template<> struct DataAccessControlerSet<>
    {
      static inline constexpr size_t ndims = 0;
      static inline constexpr bool is_valid = true;
      using accessor_tuple_t = FlatTuple<>;
    };
    template<class Dac1, class... OtherDacs>
    struct DataAccessControlerSet<Dac1,OtherDacs...>
    {
      using first_dac_t = Dac1;
      static inline constexpr size_t ndims = first_dac_t::ND;
      static inline constexpr bool is_valid = ( ... && ( OtherDacs::ND==ndims || OtherDacs::ND==0 ) );
      using accessor_tuple_t = FlatTuple<Dac1,OtherDacs...>;
    };

    template<class IS , class... DacT>
    struct FilteredDacSet
    {
      using seq_t = IS;
      using dac_set_t = DataAccessControlerSet<DacT...>;
    };

    template<class Is , size_t I> struct IndexSequenceAppend;
    template<size_t... Is , size_t I> struct IndexSequenceAppend<std::integer_sequence<std::size_t,Is...>,I> { using type = std::integer_sequence<std::size_t,Is...,I>; };
    template<class Is , size_t I> using seq_append_t = typename IndexSequenceAppend<Is,I>::type;
    
    // append a DataAccessControler to a DataAccessControlerSet
    template<class DacSetT, class DacT, bool cond, size_t I> struct DacSetAppend;
    template<class DacSetT, class DacT, size_t I> struct DacSetAppend<DacSetT,DacT,false,I> { using type = DacSetT; };
    template<class DacT, class IS , size_t I, class... T> struct DacSetAppend< FilteredDacSet<IS,T...> , DacT , true ,I> { using type = FilteredDacSet<seq_append_t<IS,I>,T...,DacT>; };
    template<class DacSetT, class DacT, bool cond, size_t I> using dac_set_append_if_t = typename DacSetAppend<DacSetT,DacT,cond,I>::type;

    template<class DacSetNdT, class DacSet0DT, class InputDacSetT > struct DacSetSplit
    {
      using dac_set_nd_t = DacSetNdT;
      using dac_set_0d_t = DacSet0DT;
    };
    template<class DacSetNdT, class DacSet0DT, class DacT, class... OtherDacs > struct DacSetSplit<DacSetNdT,DacSet0DT, DataAccessControlerSet<DacT,OtherDacs...> >
    {
      static inline constexpr bool same_dims = ( DacT::ND == DacSetNdT::dac_set_t::ndims );
      static inline constexpr size_t next_index = DacSetNdT::seq_t::size() + DacSet0DT::seq_t::size();
      using next_step_t = DacSetSplit< dac_set_append_if_t<DacSetNdT,DacT,same_dims,next_index> , dac_set_append_if_t<DacSet0DT,DacT,!same_dims,next_index> , DataAccessControlerSet<OtherDacs...> >;
      using dac_set_nd_t = typename next_step_t::dac_set_nd_t;
      using dac_set_0d_t = typename next_step_t::dac_set_0d_t;
    };

    template<class T> struct DacSetSplitter;
    template<> struct DacSetSplitter< FlatTuple<> >
    {
      using dac_set_nd_t = DataAccessControlerSet<>;
      using dac_set_nd_indices = std::index_sequence<>;
      using dac_set_0d_t = DataAccessControlerSet<>;
      using dac_set_0d_indices = std::index_sequence<>;
    };
    template<class Dac1, class... OtherDacs> struct DacSetSplitter< FlatTuple<Dac1,OtherDacs...> >
    {
      using dac_set_split_t = DacSetSplit< FilteredDacSet<std::index_sequence<0>,Dac1> , FilteredDacSet< std::index_sequence<> > , DataAccessControlerSet<OtherDacs...> >;
      using dac_set_nd_t = typename dac_set_split_t::dac_set_nd_t::dac_set_t;
      using dac_set_nd_indices = typename dac_set_split_t::dac_set_nd_t::seq_t;
      using dac_set_0d_t = typename dac_set_split_t::dac_set_0d_t::dac_set_t;
      using dac_set_0d_indices = typename dac_set_split_t::dac_set_0d_t::seq_t;
    };

    template<class AccessorTuple>
    static inline auto dac_set_nd_subset( const AccessorTuple& at )
    {
      return flat_tuple_subset( at , typename DacSetSplitter<AccessorTuple>::dac_set_nd_indices {} );
    }

    template<class AccessorTuple>
    static inline auto dac_set_0d_subset( const AccessorTuple& at )
    {
      return flat_tuple_subset( at , typename DacSetSplitter<AccessorTuple>::dac_set_0d_indices {} );
    }

    // ******************* data reference tuple from parallel task args for fulfill companion task ****************
    template<class T, bool = is_data_access_controler_v<T> > struct DacOrRawDataRefHelper
    {
      static inline constexpr typename T::reference_t data_ref(const T& x) { return x.m_ref; }
    };
    template<class T> struct DacOrRawDataRefHelper<T,false>
    {
      static inline constexpr T& data_ref(T& x) { return x; }
//      static inline constexpr const T& data_ref(const T& x) { return x; }
    };
    template<class T> static inline constexpr auto dac_or_raw_data_ref( T& x ) { return DacOrRawDataRefHelper<T>::data_ref(x); }

    // *************** find first Nd DAC in a set **************************
    template<size_t Nd, class DacSetT> struct FirstNdDac;
    template<size_t Nd> struct FirstNdDac<Nd,DataAccessControlerSet<> > { using type = FakeDataAccessControler<Nd>; using index = std::integral_constant<size_t,0> ; };
    template<size_t Nd,class Dac1, class... OtherDacs> struct FirstNdDac<Nd, DataAccessControlerSet<Dac1,OtherDacs...> >
    {
      using type = std::conditional_t< Dac1::ND==Nd , Dac1 , typename FirstNdDac<Nd,DataAccessControlerSet<OtherDacs...> >::type >;
      using index = std::conditional_t< Dac1::ND==Nd , std::integral_constant<size_t,0> , std::integral_constant<size_t,1+FirstNdDac<Nd,DataAccessControlerSet<OtherDacs...> >::index::value> > ;
    };
    template<size_t N,class... Dacs> using first_nd_dac_t = typename FirstNdDac<N,DataAccessControlerSet<Dacs...> >::type ;
    template<size_t N,class... Dacs> static inline constexpr typename FirstNdDac<N,DataAccessControlerSet<Dacs...> >::index first_nd_dac_index_v {};
        
    template<size_t N,class Dac1, class... OtherDacs>
    static inline const first_nd_dac_t<N,Dac1,OtherDacs...> & first_nd_dac( std::integral_constant<size_t,N> nd, const Dac1& acc1, const OtherDacs& ... other_accs )
    {
      if constexpr ( Dac1::ND == N ) { return acc1; }
      if constexpr ( Dac1::ND != N ) { return first_nd_dac(nd,other_accs...); }
      std::abort();
      first_nd_dac_t<N,Dac1,OtherDacs...> * never_used=nullptr; return *never_used;
    }

    template<size_t N=0> static inline FakeDataAccessControler<N> first_nd_dac( std::integral_constant<size_t,N> = std::integral_constant<size_t,N>{} ) { return {}; }

    // ***************** debugging helpers ****************************
    template<class T> struct IsConstLValueReference : public std::false_type {};
    template<class T> struct IsConstLValueReference<const T&> : public std::true_type {};
    template<class T> static inline constexpr bool is_const_lvalue_ref_v = IsConstLValueReference<T>::value;

  }
}


// *************** hash function for data access controlers **************************

#include <functional>
#include <string_view>

namespace std
{
  template<class A, class B, class C, bool D, class E>
  struct hash< onika::dac::DataAccessControler<A,B,C,D,E> >
  {
    inline size_t operator () ( const onika::dac::DataAccessControler<A,B,C,D,E> & dac) const
    {
      return std::hash<std::string_view>{}( std::string_view( (const char*) &dac , sizeof(dac) ) );
    }
  };
}

#undef __DAC_AUTO_RET_EXPR

