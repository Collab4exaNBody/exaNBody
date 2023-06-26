#pragma once

#include <onika/soatl/field_arrays.h>
#include <onika/dac/dac.h>
#include <onika/flat_tuple.h>

namespace onika
{

  namespace dac
  {

    template<class T, class = std::enable_if_t< soatl::is_field_arrays_v<T> > >
    struct SoatlSizeProxy
    {
      T& m_ref;
      inline operator size_t() const { return m_ref.size(); }
      inline SoatlSizeProxy operator = (size_t s) { m_ref.resize(s); return *this; }
    };

#   define __DAC_AUTO_RET_EXPR( ... ) -> decltype( __VA_ARGS__ ) { return __VA_ARGS__ ; }

    // identifier for container size's accessor
    struct field_array_size_t {};
    static inline constexpr field_array_size_t field_array_size = {};

    // encapsulating template to represent an optional field.
    // if one accesses an optional field and the field doesn't exist in the container, 
    // nullptr is returned rather that failing to compile
    /*
    template<class T> struct optional_field { using field_id = T; };
    template<class T> struct is_optional_field : public std::false_type {};
    template<class T> struct is_optional_field<optional_field<T> > : public std::true_type {};
    template<class T> static inline constexpr bool is_optional_field_v = is_optional_field<T>::value;
    */

    template<size_t _Align, size_t _Chunk, class _Alloc, size_t _SPCount, class... _Fields>
    struct DataSlicing< soatl::FieldArraysWithAllocator<_Align,_Chunk,_Alloc,_SPCount,_Fields...> >
    {
      using slices_t = DataSlices< field_array_size_t , _Fields ... >;
      using value_t = soatl::FieldArraysWithAllocator<_Align,_Chunk,_Alloc,_SPCount,_Fields...>;
      
      ONIKA_HOST_DEVICE_FUNC
      static inline size_t get_slice(const value_t& v , DataSliceAccess<field_array_size_t,ro_t> )
      {
        return v.size();
      }

      ONIKA_HOST_DEVICE_FUNC
      static inline SoatlSizeProxy< soatl::FieldArraysWithAllocator<_Align,_Chunk,_Alloc,_SPCount,_Fields...> > get_slice(value_t& v , DataSliceAccess<field_array_size_t,rw_t> )
      {
        return {v};
      }

      template<class _id, class = std::enable_if_t< soatl::find_index_of_id_v<_id,_Fields...> != soatl::bad_field_index /*|| is_optional_field_v<_id> */ > >
      ONIKA_HOST_DEVICE_FUNC
      static inline
      typename soatl::FieldId<_id>::value_type const * __restrict__ 
      get_slice(const value_t& v , DataSliceAccess<_id,ro_t> )
      {
        return v[soatl::FieldId<_id>{}];
      }

      template<class _id, class = std::enable_if_t< soatl::find_index_of_id_v<_id,_Fields...> != soatl::bad_field_index > >
      ONIKA_HOST_DEVICE_FUNC
      static inline
      typename soatl::FieldId<_id>::value_type * __restrict__ 
      get_slice(value_t& v , DataSliceAccess<_id,rw_t> )
      {
        return v[soatl::FieldId<_id>{}];
      }

    };

    // recover a field pointer from an accessor tuple, given the stencil that generated it and the field identifier var
    template<class C, class Es, size_t S, class _id, class... T> 
    static inline auto field_pointer_from_accessor( const FlatTuple<T...>& tp , dac::Stencil<C,Es,S> , soatl::FieldId<_id> )
    {
      static constexpr ssize_t index = C::ro_rw_slices_t::slices_t::slice_index_weak( _id{} );
      static constexpr size_t nb_ro_slices = C::ro_rw_slices_t::nb_ro_slices;
      //static constexpr size_t nb_slices = C::ro_rw_slices_t::nb_slices;
      using rtype = std::conditional_t< index>=0 ,
                      std::conditional_t< index < ssize_t(nb_ro_slices) ,
                          typename soatl::FieldId<_id>::value_type const * __restrict__ ,
                          typename soatl::FieldId<_id>::value_type * __restrict__ > ,
                      std::nullptr_t >;
      rtype r{};
      if constexpr ( index>=0 ) { r = tp. template get_nth_const<index> (); }
      return r;
    }

/*
    template<class _FS, size_t _C >
    struct DataDecompositionTraits< ::ustamp::Grid<_FS,_C> >
    {
      using IJK = ::ustamp::IJK;
      using value_t = ::ustamp::Grid<_FS,_C>;
      using item_t = typename value_t::CellParticles;
      using slices_t = typename DataSlicing<item_t>::slices_t;

      static constexpr size_t ND = 3;
      using item_coord_t = std::array<size_t,ND>;

      static inline item_coord_t size(const value_t& v)
      {
        auto dims = v.dimension();
        return { dims.i , dims.j , dims.k };
      }
      static inline size_t count(const value_t& v) { return v.number_of_cells(); }

      static inline const item_t& get(const value_t& v , const item_coord_t& c) { return v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}); }
      static inline void set(value_t& v, const item_t& n, const item_coord_t& c) { v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) = n; }

      static inline void* slice_addr( value_t& v , const item_coord_t& c , unsigned int i )
      { return DataSliceAddress<item_t>::slice_address( & v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) , i); }

      template<class S>
      static inline auto get_slice_ro(const value_t& v , const item_coord_t& c , S )
      __DAC_AUTO_RET_EXPR( DataSlicing<item_t>::get_slice_ro( v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) ,S{}) )

      template<class S>
      static inline auto get_slice_rw(value_t& v , const item_coord_t& c , S )
      __DAC_AUTO_RET_EXPR( DataSlicing<item_t>::get_slice_rw( v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}),S{}) )

      template<class... S>
      static inline auto get_slices_ro(const value_t& v, const item_coord_t& c, data_slices_subset_t<item_t,S...> )
      __DAC_AUTO_RET_EXPR( data_slices_subset_t<item_t,S...>::get_slices_ro( v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) ) )

      template<class... S>
      static inline auto get_slices_rw(value_t& v, const item_coord_t& c, data_slices_subset_t<item_t,S...> )
      __DAC_AUTO_RET_EXPR( data_slices_subset_t<item_t,S...>::get_slices_rw( v.cell(IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) ) )
    };
*/

#   undef __DAC_AUTO_RET_EXPR



  }

}

