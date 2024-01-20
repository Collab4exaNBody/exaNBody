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

namespace exanb
{

  template<class CellsT, class FuncT> struct ExternalCellParticleFieldAccessor;
/*
    {
      FuncT m_func;
      using value_type = decltype(0,0,m_func(CellsT{}));
      static const char* short_name() { return "external"; }
      static const char* name() { return "external"; }
    };
*/

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
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( const onika::soatl::FieldId<field_id>& f ) const
    {
      return m_cells[m_cell_index][f];
    }

    template<class FuncT , class... fids>
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( const onika::soatl::FieldCombiner<FuncT,fids...>& f ) const
    {
      return m_cells[m_cell_index][f];
    }

    template<class FuncT>
    ONIKA_HOST_DEVICE_FUNC inline GridParticleFieldAccessor2<CellsT, ExternalCellParticleFieldAccessor<CellsT,FuncT> > operator [] ( const ExternalCellParticleFieldAccessor<CellsT,FuncT>& f ) const
    {
      return { m_cells , m_cell_index , f };
    }    
  };

  template<class CellsT, class FieldAccT>
  struct GridParticleFieldAccessor2
  {
    CellsT m_cells;    
    const size_t m_cell_index;
    const FieldAccT& m_acc_func;
    ONIKA_HOST_DEVICE_FUNC inline auto operator [] ( size_t particle_index ) const
    {
      return m_acc_func( m_cell_index , particle_index , m_cells );
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
    ONIKA_HOST_DEVICE_FUNC inline auto get(size_t cell_i, size_t p_i, const onika::soatl::FieldId<field_id>& f ) const
    {
      return m_cells[cell_i][f][p_i];
    }

    template<class FuncT , class... fids>
    ONIKA_HOST_DEVICE_FUNC inline auto get(size_t cell_i, size_t p_i, const onika::soatl::FieldCombiner<FuncT,fids...>& f ) const
    {
      return m_cells[cell_i][f][p_i];
    }

    template<class FuncT>
    ONIKA_HOST_DEVICE_FUNC inline auto get(size_t cell_i, size_t p_i, const ExternalCellParticleFieldAccessor<CellsT,FuncT>& f ) const
    {
      return f.m_func( cell_i , p_i , m_cells );
    }    
  };



  template<class T>
  static inline const GridParticleFieldAccessor<T>& convert_cell_array_ptr_to_accessor( const GridParticleFieldAccessor<T>& cells )
  {
    return cells;
  };
  
  template<class T>
  static inline GridParticleFieldAccessor<T * const> convert_cell_array_ptr_to_accessor( T * const cells )
  {
    return { cells };
  };

//template<size_t A, size_t C, typename Al, size_t N, typename... Ids >

} // end of namespace exanb

