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

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS constant
#include <exanb/core/basic_types_def.h>
#include <exanb/core/field_set_proto.h>

#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/soatl/field_tuple.h>
#include <functional>

// debug configuration
//#include "exanb/debug/debug_particle_id.h"

namespace exanb
{

  struct NoExtraStorage{};  

  // Compute buffer post processing API
  struct DefaultComputePairBufferAppendFunc
  {  
    template<class ComputeBufferT, class FieldArraysT, class NbhDataT=double>
    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void operator () (ComputeBufferT& tab, const Vec3d& dr, double d2,
                      FieldArraysT cells, size_t cell_b, size_t p_b,
                      const NbhDataT& nbh_data = 1.0 ) const noexcept
    {
      tab.d2[tab.count] = d2;
      tab.drx[tab.count] = dr.x;
      tab.dry[tab.count] = dr.y;
      tab.drz[tab.count] = dr.z;
      tab.nbh.set( tab.count , cell_b, p_b );
      tab.nbh_data.set( tab.count , nbh_data );
      ++ tab.count;
    }
  };

  struct NullComputePairBufferAppendFunc
  {  
    template<class ComputeBufferT, class FieldArraysT, class NbhDataT>
    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void operator () (ComputeBufferT&, const Vec3d&, double, FieldArraysT, size_t, size_t, const NbhDataT& ) const noexcept {}
  };

  template<bool UseWeights=false , size_t _MaxNeighbors=exanb::MAX_PARTICLE_NEIGHBORS >
  struct ComputePairBuffer2Weights
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void set( size_t , double ) noexcept { }
    ONIKA_HOST_DEVICE_FUNC static inline constexpr double get( size_t ) noexcept { return 1.0; }
    ONIKA_HOST_DEVICE_FUNC inline constexpr double operator [] ( size_t ) noexcept { return 1.0; } // for backward compatibility
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void copy(size_t , size_t ) noexcept { }
  };
  template<size_t _MaxNeighbors>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2Weights<true,_MaxNeighbors>
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    alignas(onika::memory::DEFAULT_ALIGNMENT) double weight[MaxNeighbors];  // weight to apply on pair interaction
    ONIKA_HOST_DEVICE_FUNC inline void set( size_t i, double w ) noexcept { weight[i] = w; }
    ONIKA_HOST_DEVICE_FUNC inline double get( size_t i ) const noexcept { return weight[i]; }
    ONIKA_HOST_DEVICE_FUNC inline double operator [] ( size_t i) const noexcept { return weight[i]; } // for backward compatibility
    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t src, size_t dst) noexcept { weight[dst]=weight[src]; }
  };

  template<bool Symmetric=false , size_t _MaxNeighbors=exanb::MAX_PARTICLE_NEIGHBORS >
  struct ComputePairBuffer2Nbh
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void set( size_t, size_t, size_t ) noexcept { }
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void get( size_t, size_t&, size_t&) noexcept { }
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void copy(size_t , size_t) noexcept { }
  };
  template<size_t _MaxNeighbors>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2Nbh<true,_MaxNeighbors>
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    alignas(onika::memory::DEFAULT_ALIGNMENT) std::pair<uint32_t,uint32_t> nbh[MaxNeighbors];   // encodes neighbor's cell index (in grid) and particle index (in cell)
    ONIKA_HOST_DEVICE_FUNC inline void set( size_t i , size_t c, size_t p ) noexcept {  nbh[i].first=c; nbh[i].second=p; }
    ONIKA_HOST_DEVICE_FUNC inline void get( size_t i , size_t& c, size_t& p ) const noexcept { c=nbh[i].first; p=nbh[i].second; }
    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t src, size_t dst) noexcept { nbh[dst].first=nbh[src].first; nbh[dst].second=nbh[src].second; }
  };

  template<size_t _MaxNeighbors , class FieldSetT > struct ComputePairBuffer2NbhFields;

  template<size_t _MaxNeighbors , class... field_ids >
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2NbhFields<_MaxNeighbors , FieldSet<field_ids ...> >
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    static inline constexpr bool HasFields = true;
    using FieldTupleT = onika::soatl::FieldTuple< field_ids ... >;
    alignas(onika::memory::DEFAULT_ALIGNMENT) onika::soatl::FieldTuple< field_ids ... > m_fields[MaxNeighbors];
    ONIKA_HOST_DEVICE_FUNC inline FieldTupleT& operator [] (size_t i ) noexcept { return m_fields[i]; }
    ONIKA_HOST_DEVICE_FUNC inline const FieldTupleT& operator [] (size_t i ) const noexcept { return m_fields[i]; }
    ONIKA_HOST_DEVICE_FUNC inline void set(size_t i , const FieldTupleT& tp ) noexcept { m_fields[i] = tp; }
    ONIKA_HOST_DEVICE_FUNC inline const FieldTupleT& get(size_t i ) const noexcept { return m_fields[i]; }
    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t src, size_t dst) noexcept { m_fields[dst] = m_fields[src]; }
  };

  template<size_t _MaxNeighbors>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2NbhFields<_MaxNeighbors , FieldSet<> >
  {
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    static inline constexpr bool HasFields = false;
    using FieldTupleT = onika::soatl::FieldTuple<>;
    ONIKA_HOST_DEVICE_FUNC inline void set(size_t,FieldTupleT) noexcept {}
    ONIKA_HOST_DEVICE_FUNC inline FieldTupleT get(size_t) const noexcept { return {}; }
    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t,size_t) noexcept {}
  };

  template<
    bool                        UserNbhData                 = false,
    bool                        UseNeighbors                = false,
    class                       _ExtendedStorage            = NoExtraStorage,
    class                       _BufferProcessFunc          = DefaultComputePairBufferAppendFunc,
    size_t                      _MaxNeighbors               = exanb::MAX_PARTICLE_NEIGHBORS,
    template<bool,size_t> class _UserNeighborDataBufferTmpl = ComputePairBuffer2Weights ,
    class                       _NbhFieldSetT               = FieldSet<>
    >
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2
  {
    using ExtendedStorage = _ExtendedStorage;
    using BufferProcessFunc = _BufferProcessFunc;
    template<bool b, size_t n> using UserNeighborDataBufferTmpl = _UserNeighborDataBufferTmpl<b,n>;
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
    using NbhFieldSet = _NbhFieldSetT;
    static inline constexpr NbhFieldSet nbh_field_set = {};
    using CPBufNbhFields = ComputePairBuffer2NbhFields<MaxNeighbors,NbhFieldSet>;
    using NbhFieldTuple = typename CPBufNbhFields::FieldTupleT;
  
    alignas(onika::memory::DEFAULT_ALIGNMENT) double drx[MaxNeighbors];     // neighbor's relative position x to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double dry[MaxNeighbors];     // neighbor's relative position y to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double drz[MaxNeighbors];     // neighbor's relative position z to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double d2[MaxNeighbors];      // squared distance between reference particle and neighbor
    ExtendedStorage ext;    
    UserNeighborDataBufferTmpl<UserNbhData,MaxNeighbors> nbh_data;
    ComputePairBuffer2Nbh<UseNeighbors,MaxNeighbors> nbh;
    CPBufNbhFields nbh_pt;
    uint64_t cell; // current cell
    uint32_t part; // current particle
    int32_t count; // number of neighbors stored in buffer
    uint32_t ta; // type of particle A (current particle)
    uint32_t tb; // type of particle B (neighbor particles type)
    BufferProcessFunc process_neighbor;
    
    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t src, size_t dst) noexcept
    {
      drx[dst] = drx[src];
      dry[dst] = dry[src];
      drz[dst] = drz[src];
      d2[dst] = d2[src];
      nbh_data.copy(src,dst);
      nbh.copy(src,dst);
      nbh_pt.copy(src,dst);
    }

    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void check_buffer_overflow()
    {
#     ifndef NDEBUG
      if( ssize_t(count) >= ssize_t(MaxNeighbors) )
      {
        printf("Compute buffer overflow : max capacity (%d) reached (count=%d,cell=%llu,part=%d)\n" , int(MaxNeighbors) , int(count) , static_cast<unsigned long long>(cell) , int(part) );
        ONIKA_CU_ABORT();
      }
#     endif
    }
  };

  template<class _ExtendedStorage>
  struct ComputeContextNoBuffer
  {
    using ExtendedStorage = _ExtendedStorage;
    ExtendedStorage ext;
  };


  template< class _NbhFieldSetT , size_t _MaxNeighbors = exanb::MAX_PARTICLE_NEIGHBORS , bool UserNbhData=false >
  using SimpleNbhComputeBuffer = ComputePairBuffer2< UserNbhData, false, NoExtraStorage, DefaultComputePairBufferAppendFunc, _MaxNeighbors, ComputePairBuffer2Weights, _NbhFieldSetT >;

  // this allows to store the ComputeBufferType and a function to initialize it
  struct CPBufNullInit
  {
    template<class CPBufT>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (CPBufT&) const {}
  };

  template<class CPBufT, class InitFuncT = CPBufNullInit >
  struct ComputePairBufferFactory
  {
    using ComputePairBuffer = CPBufT;
    InitFuncT m_init_func;
    ONIKA_HOST_DEVICE_FUNC inline void init(ComputePairBuffer& buf) const { m_init_func(buf); }
  };

  template<class CPBufT>
  static inline ComputePairBufferFactory< CPBufT, CPBufNullInit >
  make_compute_pair_buffer()
  {
    return ComputePairBufferFactory< CPBufT, CPBufNullInit > { {} };
  }

  template<class CPBufT>
  static inline ComputePairBufferFactory< CPBufT, std::function<void(CPBufT&)> >
  make_compute_pair_buffer( const std::function<void(CPBufT&)> & init_func )
  {
    return ComputePairBufferFactory< CPBufT, std::function<void(CPBufT&)> > { init_func };
  }

  static inline constexpr ComputePairBufferFactory< ComputePairBuffer2<> > make_default_pair_buffer() { return {}; }

  template<class ExtStorageT = NoExtraStorage >
  static inline constexpr ComputePairBufferFactory< ComputeContextNoBuffer<ExtStorageT> > make_empty_pair_buffer() { return {}; }
}

