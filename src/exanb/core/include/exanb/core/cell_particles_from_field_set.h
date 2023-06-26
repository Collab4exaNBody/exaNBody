#pragma once

#include <exanb/field_sets.h>
#include <onika/memory/allocator.h>
#include <onika/soatl/field_arrays.h>

namespace exanb
{

  template<typename field_set> struct CellParticlesFromFieldSet;
  template<typename... field_ids > struct CellParticlesFromFieldSet< FieldSet<field_ids...> >
  {
    static constexpr size_t StoredPointerCount = std::min( size_t(XSTAMP_FIELD_ARRAYS_STORE_COUNT) , size_t(sizeof...(field_ids)+3) );
    using CellParticlesAllocator = onika::soatl::PackedFieldArraysAllocatorImpl< onika::memory::DefaultAllocator, onika::memory::DEFAULT_ALIGNMENT, onika::memory::DEFAULT_CHUNK_SIZE, field::_rx,field::_ry,field::_rz,field_ids... > ;
    using type = ::onika::soatl::FieldArraysWithAllocator< onika::memory::DEFAULT_ALIGNMENT
                                                               , onika::memory::DEFAULT_CHUNK_SIZE
                                                               , CellParticlesAllocator
                                                               , StoredPointerCount
                                                               , field::_rx,field::_ry,field::_rz,field_ids...>;
  };

  template<class FS> using cell_particles_from_field_set_t = typename CellParticlesFromFieldSet<FS>::type;

}

