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

