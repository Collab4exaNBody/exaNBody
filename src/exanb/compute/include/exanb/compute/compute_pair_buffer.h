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

#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <functional>

// debug configuration
//#include "exanb/debug/debug_particle_id.h"

namespace exanb
{

  struct NoExtraStorage{};  

  // Compute buffer post processing API
  struct DefaultComputePairBufferAppendFunc
  {  
    template<class ComputeBufferT, class FieldArraysT, class NbhDataT, bool ForceCheck=false >
    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void operator () (ComputeBufferT& tab, const Vec3d& dr, double d2,
                                                    const FieldArraysT * cells, size_t cell_b, size_t p_b,
                                                    const NbhDataT& nbh_data,
                                                    std::integral_constant<bool,ForceCheck> force_check_overflow = {} ) const noexcept
    {
      tab.check_buffer_overflow( force_check_overflow );
      tab.d2[tab.count] = d2;
      tab.drx[tab.count] = dr.x;
      tab.dry[tab.count] = dr.y;
      tab.drz[tab.count] = dr.z;
      tab.nbh.set( tab.count , cell_b, p_b );
      tab.nbh_data.set( tab.count , nbh_data );
      ++ tab.count;
    }
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

  template<
    bool UserNbhData=false,
    bool UseNeighbors=false,
    class _ExtendedStorage=NoExtraStorage,
    class _BufferProcessFunc=DefaultComputePairBufferAppendFunc,
    size_t _MaxNeighbors=exanb::MAX_PARTICLE_NEIGHBORS,
    template<bool,size_t> class _UserNeighborDataBufferTmpl = ComputePairBuffer2Weights
    >
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) ComputePairBuffer2
  {
    using ExtendedStorage = _ExtendedStorage;
    using BufferProcessFunc = _BufferProcessFunc;
    template<bool b, size_t n> using UserNeighborDataBufferTmpl = _UserNeighborDataBufferTmpl<b,n>;
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;
  
    alignas(onika::memory::DEFAULT_ALIGNMENT) double drx[MaxNeighbors];     // neighbor's relative position x to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double dry[MaxNeighbors];     // neighbor's relative position y to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double drz[MaxNeighbors];     // neighbor's relative position z to reference particle
    alignas(onika::memory::DEFAULT_ALIGNMENT) double d2[MaxNeighbors];      // squared distance between reference particle and neighbor
    ExtendedStorage ext;    
    UserNeighborDataBufferTmpl<UserNbhData,MaxNeighbors> nbh_data;
    ComputePairBuffer2Nbh<UseNeighbors,MaxNeighbors> nbh;
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
    }

    template<bool ForceCheck=false>
    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void check_buffer_overflow( std::integral_constant<bool,ForceCheck> = {} )
    {
#     ifndef NDEBUG
      static constexpr bool CheckBufferOverflow = true;
#     else
      static constexpr bool CheckBufferOverflow = ForceCheck;
#     endif
      if constexpr ( CheckBufferOverflow )
      {
        if( ssize_t(count) >= ssize_t(MaxNeighbors) )
        {
          printf("Compute buffer overflow : max capacity (%d) reached (count=%d,cell=%llu,part=%d)\n" , int(MaxNeighbors) , int(count) , static_cast<unsigned long long>(cell) , int(part) );
          ONIKA_CU_ABORT();
        }
      }
    }
    
  };

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

}

