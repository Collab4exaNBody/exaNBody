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

  template<class CellsT> struct GridParticleFieldAccessor;
  template<class CellsT> struct GridParticleFieldAccessor1;
  template<class CellsT, class FieldAccT> struct GridParticleFieldAccessor2;

  template<class CellsT>
  struct GridParticleFieldAccessor1
  {
    CellsT m_cells;    
    const size_t m_cell_index;    

    ONIKA_HOST_DEVICE_FUNC inline auto size() const
    {
      return m_cells[m_cell_index].size();
    }

    template<class field_id>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( onika::soatl::FieldId<field_id> ) const
    {
      return m_cells[m_cell_index][ onika::soatl::FieldId<field_id>{} ];
    }

    template<class FuncT , class... fids>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( const onika::soatl::FieldCombiner<FuncT,fids...>& f ) const
    {
      return m_cells[m_cell_index][f];
    }

    template<class FuncT, class FieldIdT>
    ONIKA_HOST_DEVICE_FUNC inline GridParticleFieldAccessor2<CellsT, ExternalCellParticleFieldAccessor<FuncT,FieldIdT> > operator [] ( const ExternalCellParticleFieldAccessor<FuncT,FieldIdT>& f ) const
    {
      return { m_cells , m_cell_index , f };
    }    
  };

  template<class CellsT, class FieldAccT>
  struct GridParticleFieldAccessor2
  {
    CellsT m_cells;    
    const size_t m_cell_index;
    const FieldAccT& m_field_acc;
    ONIKA_HOST_DEVICE_FUNC inline typename FieldAccT::reference_t operator [] ( size_t particle_index ) const
    {
      return m_field_acc.m_func( m_cell_index , particle_index , m_cells );
    }
  };

  template<class CellsT>
  struct GridParticleFieldAccessor
  {
    CellsT m_cells;    

    ONIKA_HOST_DEVICE_FUNC inline GridParticleFieldAccessor1<CellsT> operator [] (size_t cell_i) const
    {
      return { m_cells , cell_i };
    }

    template<class field_id>
    ONIKA_HOST_DEVICE_FUNC inline auto get(size_t cell_i, size_t p_i, onika::soatl::FieldId<field_id> ) const
    {
      return m_cells[cell_i][ onika::soatl::FieldId<field_id>{} ][p_i];
    }

    template<class FuncT , class... fids>
    ONIKA_HOST_DEVICE_FUNC inline auto get(size_t cell_i, size_t p_i, const onika::soatl::FieldCombiner<FuncT,fids...>& f ) const
    {
      return m_cells[cell_i][f][p_i];
    }

    template<class FuncT, class FieldIdT>
    ONIKA_HOST_DEVICE_FUNC inline typename FuncT::reference_t get(size_t cell_i, size_t p_i, const ExternalCellParticleFieldAccessor<FuncT,FieldIdT>& f ) const
    {
      return f.m_func( cell_i , p_i , m_cells );
    }    
  };

  template<class FieldIdT, bool ConstAccessor=false > struct GridExternalFieldFlatArrayAccessor
  {
    using value_type = typename FieldIdT::value_type;
    using reference_t = std::conditional_t<ConstAccessor, const value_type & , value_type & >;
    using unmutable_pointer_t = std::conditional_t<ConstAccessor, value_type const * const __restrict__ , value_type * const __restrict__ >;
    
    size_t const * const m_cell_particle_offset = nullptr;
    unmutable_pointer_t m_data_array = nullptr;
    
    template<class CellsT>
    ONIKA_HOST_DEVICE_FUNC
    inline reference_t operator () ( size_t cell_i, size_t p_i , CellsT ) const
    {
      return m_data_array[ m_cell_particle_offset[cell_i] + p_i ];
    }
  };

  template<class GridT, class FieldIdT>
  static inline auto make_external_field_flat_array_accessor( const GridT& grid, typename FieldIdT::value_type * const array , FieldIdT)
  {
    using ExternalFieldT = ExternalCellParticleFieldAccessor< GridExternalFieldFlatArrayAccessor<FieldIdT,false> , FieldIdT >;
    return ExternalFieldT{ grid.cell_particle_offset_data() , array };
  }

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

  template<class... field_ids , class... FieldAccessorT>
  static inline auto make_field_tuple_from_field_set( FieldSet<field_ids...> , const FieldAccessorT& ... f )
  {
    return onika::FlatTuple< onika::soatl::FieldId<field_ids> ... , FieldAccessorT ... > { onika::soatl::FieldId<field_ids>{} ... , f ... };
  }
  
} // end of namespace exanb

