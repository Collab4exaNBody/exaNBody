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

#include <onika/cuda/cuda.h>
#include <onika/soatl/field_id.h>
#include <onika/soatl/field_combiner.h>
#include <onika/flat_tuple.h>

namespace exanb
{

  template<class FuncT, class FieldIdT> struct ExternalCellParticleFieldAccessor
  {
    FuncT m_func;
    using field_id = FieldIdT;
    using Id = typename FieldIdT::Id;
    using value_type = typename FuncT::value_type;
    using reference_t = typename FuncT::reference_t;
    // using unmutable_pointer_t = typename FuncT::unmutable_pointer_t;

    static inline constexpr const char* short_name() { return field_id::short_name(); }
    static inline constexpr const char* name() { return field_id::name(); }
  };

  template<class FieldAccTupleT> struct FieldAccessorTupleHasExternalFields;
  template<>
  struct FieldAccessorTupleHasExternalFields< onika::FlatTuple<> >
  {
    static inline constexpr bool value = false;
  };
  template<class FieldAccT0, class... FieldAccT >
  struct FieldAccessorTupleHasExternalFields< onika::FlatTuple<FieldAccT0,FieldAccT...> >
  {
    static inline constexpr bool value = FieldAccessorTupleHasExternalFields< onika::FlatTuple<FieldAccT...> >::value ;
  };
  template<class FuncT, class FieldIdT, class... FieldAccT >
  struct FieldAccessorTupleHasExternalFields< onika::FlatTuple< ExternalCellParticleFieldAccessor<FuncT,FieldIdT> , FieldAccT...> >
  {
    static inline constexpr bool value = true;
  };
  template<class T> static inline constexpr bool field_tuple_has_external_fields_v = FieldAccessorTupleHasExternalFields<T>::value;

  template<class GridT, bool ConstAccessor> struct GridParticleFieldAccessor;
  template<class GridT, bool ConstAccessor> struct GridParticleFieldAccessor1;

  // second stage accessor, returns a pointer to a field for a given cell index
  template<class GridT, bool ConstAccessor>
  struct GridParticleFieldAccessor1
  {
    using CellsT = std::conditional_t< ConstAccessor , typename GridT::CellParticles const * const , typename GridT::CellParticles * const>;
    using OptCellsT = std::conditional_t< ConstAccessor , typename GridT::optional_const_pointers_t , typename GridT::optional_pointers_t >;
    using optional_field_sets_t = typename GridT::optional_field_sets_t;
    CellsT m_cells = nullptr;  
    OptCellsT m_optional_cells = {}; // tuple of pointer to optional cell arrays
    const size_t m_cell_index;

    ONIKA_HOST_DEVICE_FUNC inline auto size() const
    {
      return m_cells[m_cell_index].size();
    }

    template<class... fids>
    ONIKA_HOST_DEVICE_FUNC inline void set_tuple( size_t i, const onika::soatl::FieldTuple<fids...> & tp ) const
    {
      ( ... , ( (*this)[m_cell_index][onika::soatl::FieldId<fids>()][i] = tp[ onika::soatl::FieldId<fids>() ] ) ); // may crash if optional field not allocated
    }
    
    template<class... fids>
    ONIKA_HOST_DEVICE_FUNC inline void write_tuple( size_t i, const onika::soatl::FieldTuple<fids...> & tp ) const
    {
      ( ... , ( (*this)[m_cell_index][onika::soatl::FieldId<fids>()][i] = tp[ onika::soatl::FieldId<fids>() ] ) );
    }

    template<class field_id>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( onika::soatl::FieldId<field_id> ) const
    {
      using OptPkgIndexT = GridT::FindOptionalPackageIndex< typename FieldIdT::Id , optional_field_sets_t > ;
      static constexpr int opt_pkg_idx = OptPkgIndexT::value;
      if constexpr ( opt_pkg_idx >= 0 )
      {
        auto cell_ptr = m_optional_cells.get( onika::tuple_index<opt_pkg_idx> );
        return (cell_ptr!=nullptr) ? cell_ptr[m_cell_index][ onika::soatl::FieldId<field_id>{} ] : nullptr;
      }
      else
      {
        return m_cells[m_cell_index][ onika::soatl::FieldId<field_id>{} ];
      }
    }

    // no optional field combiner is possible yet
    template<class FuncT , class... fids>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( const onika::soatl::FieldCombiner<FuncT,fids...>& f ) const
    {
      return m_cells[m_cell_index][f];
    }

    template<class FuncT, class FieldIdT>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( const ExternalCellParticleFieldAccessor<FuncT,FieldIdT>& f ) const
    {
      return f.m_func( m_cell_index );
    }
  };

  template<class GridT, bool ConstAccessor>
  struct GridParticleFieldAccessor
  {
    using CellsT = std::conditional_t< ConstAccessor , typename GridT::CellParticles const * const , typename GridT::CellParticles * const >;
    using OptCellsT = std::conditional_t< ConstAccessor , typename GridT::optional_const_pointers_t , typename GridT::optional_pointers_t >;
    CellsT m_cells = nullptr;    
    OptCellsT m_optional_cells = {}; // should be a tuple of pointer to optiona cell arrays
    ONIKA_HOST_DEVICE_FUNC inline GridParticleFieldAccessor1<GridT,ConstAccessor> operator [] (size_t cell_i) const
    {
      return { m_cells , m_optional_cells, cell_i };
    }
  };

  /*
   * Allows a particle field, stored as a flat 1D array, to be used as a particle field through the ExternalCellParticleFieldAccessor envelop
   */
  template<class FieldIdT, bool ConstAccessor=false > struct GridExternalFieldFlatArrayAccessor
  {
    using value_type = typename FieldIdT::value_type;
    using reference_t = std::conditional_t<ConstAccessor, const value_type & , value_type & >;
    using pointer_t = std::conditional_t<ConstAccessor, value_type const * __restrict__ , value_type * __restrict__ >;
    size_t const * m_cell_particle_offset = nullptr;
    pointer_t m_data_array = nullptr;
    ONIKA_HOST_DEVICE_FUNC inline pointer_t operator () ( size_t cell_i ) const
    {
      return m_data_array + m_cell_particle_offset[cell_i] ;
    }
  };

  // convinience function to build up a flat array acessor from a writable array
  template<class GridT, class FieldIdT>
  static inline auto make_external_field_flat_array_accessor( const GridT& grid, typename FieldIdT::value_type * const array , FieldIdT)
  {
    using ExternalFieldT = ExternalCellParticleFieldAccessor< GridExternalFieldFlatArrayAccessor<FieldIdT,false> , FieldIdT >;
    return ExternalFieldT{ grid.cell_particle_offset_data() , array };
  }

  // convinience function to build up a flat array acessor from a read-only array
  template<class GridT, class FieldIdT>
  static inline auto make_external_field_flat_array_accessor( const GridT& grid, typename FieldIdT::value_type const * const array , FieldIdT)
  {
    using ExternalFieldT = ExternalCellParticleFieldAccessor< GridExternalFieldFlatArrayAccessor<FieldIdT,true> , FieldIdT >;
    return ExternalFieldT{ grid.cell_particle_offset_data() , array };
  }

  // convert FieldSet to tuple of Field Accessor
  template<class FieldSetT> struct FieldAccessorTupleFromFieldSet;
  template<class... field_ids> struct FieldAccessorTupleFromFieldSet< FieldSet<field_ids...> > { using type = onika::FlatTuple< onika::soatl::FieldId<field_ids> ... >; };
  template<class FieldSetT> using field_accessor_tuple_from_field_set_t = typename FieldAccessorTupleFromFieldSet<FieldSetT>::type;

  // assembles fields contained in a FieldSet and other fields to a Field Accessor Tuple
  template<class... field_ids , class... FieldAccessorT>
  static inline auto make_field_tuple_from_field_set( FieldSet<field_ids...> , const FieldAccessorT& ... f )
  {
    return onika::FlatTuple< onika::soatl::FieldId<field_ids> ... , FieldAccessorT ... > { onika::soatl::FieldId<field_ids>{} ... , f ... };
  }
  
  // extract FieldId's low level ids from miscellaneous accessors, and build a FieldSet
  template<class FieldSetT> struct FieldAccessorTupleToFieldSet;
  template<class... FieldAccT> struct FieldAccessorTupleToFieldSet< onika::FlatTuple< FieldAccT... > > { using type = FieldSet< typename FieldAccT::Id ... >; };
  template<class FieldSetT> using field_accessor_tuple_to_field_set_t = typename FieldAccessorTupleToFieldSet<FieldSetT>::type;
  
  template<class FieldAccT> using field_id_fom_acc_t = onika::soatl::FieldId< typename FieldAccT::Id >;
  template<class FieldAccT> static inline constexpr field_id_fom_acc_t<FieldAccT> field_id_fom_acc_v = {};
  
  template<class StreamT, class FieldAccTupleT, size_t... I>
  static inline StreamT& print_field_tuple(StreamT& out, const FieldAccTupleT& tp , std::index_sequence<I...> )
  {
    int s=0;
    ( ... , ( out << (((s++)==0)?"":",") << tp.get(onika::tuple_index_t<I>{}).short_name() ) ) ;
    return out;
  }
  template<class StreamT, class FieldAccTupleT>
  static inline StreamT& print_field_tuple(StreamT& out, const FieldAccTupleT& tp)
  {
    return print_field_tuple(out,tp,std::make_index_sequence< onika::tuple_size_const_v<FieldAccTupleT> >{});
  }
  
} // end of namespace exanb

