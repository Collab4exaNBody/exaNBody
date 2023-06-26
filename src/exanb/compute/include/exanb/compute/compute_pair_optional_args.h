#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/thread.h>

#include <exanb/core/compact_grid_pair_weights.h>
#include <exanb/core/xform.h>

#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/memory/allocator.h>

namespace exanb
{

  // ======================= new optional args structure ===========================
  struct ComputePairTrivialCellFiltering
  {
    ONIKA_HOST_DEVICE_FUNC inline bool operator () (size_t,const IJK&) const { return true; }
  };
  
  struct ComputePairTrivialParticleFiltering
  {
    ONIKA_HOST_DEVICE_FUNC inline bool operator () (size_t,size_t) const { return true; }
  };

  template<class NbhIteratorT, class NbhDataIteratorT, class TransformT, class GridParticleLocksT
         , class CellFilteringFuncT = ComputePairTrivialCellFiltering
         , class ParticleFilteringFuncT = ComputePairTrivialParticleFiltering >
  struct ComputePairOptionalArgs2
  {
    using nbh_iterator_t = NbhIteratorT;
    using nbh_data_iterator_t = NbhDataIteratorT;
    using transform_t = TransformT;
    using particle_locks_t = GridParticleLocksT;
    
    nbh_iterator_t nbh;
    nbh_data_iterator_t nbh_data;
    transform_t xform;
    particle_locks_t locks;
    CellFilteringFuncT cell_filter;
    ParticleFilteringFuncT particle_filter;
  };

  // ---- optional weights ------
  
  struct ComputePairWeightIterator
  {
    static inline constexpr bool c_has_nbh_data = true;
    static inline constexpr double c_default_value = 1.0;

    const std::vector< std::vector<double> >& m_weights;
    const std::vector< unsigned int > m_offset;

    struct PairWeightIteratorCtx {};
    ONIKA_HOST_DEVICE_FUNC static inline PairWeightIteratorCtx make_ctx() { return {}; }

    inline double get(size_t cell_i, size_t p_i, size_t p_nbh_index, PairWeightIteratorCtx&) const noexcept
    {
      // obsolete, m_offset has to be a reference to something set correctly
      std::abort();
      return m_weights[cell_i][ m_offset[p_i] + p_nbh_index ];
    }
  };

  struct CompactPairWeightIterator
  {
    static inline constexpr bool c_has_nbh_data = true;
    static inline constexpr double c_default_value = 1.0;

    const CompactCellParticlePairWeights* m_cell_weights = nullptr;
    struct PairWeightIteratorCtx {};
    
    ONIKA_HOST_DEVICE_FUNC static inline PairWeightIteratorCtx make_ctx() { return {}; }
    
    ONIKA_HOST_DEVICE_FUNC inline double get(size_t cell_i, size_t p_i, size_t p_nbh_index, PairWeightIteratorCtx&) const noexcept
    {
      return m_cell_weights[cell_i].pair_weight( p_i , p_nbh_index );
    }
  };

  struct ComputePairDualWeightIterator
  {
    static inline constexpr bool c_has_nbh_data = true;
    static inline constexpr double c_default_value = 1.0;

    const std::vector< std::vector<double> >& m_p_weights;
    const std::vector< std::vector<double> >& m_d_weights;

    const std::vector< unsigned int > m_p_offset;
    const std::vector< unsigned int > m_d_offset;

    struct PairWeightIteratorCtx {};
    ONIKA_HOST_DEVICE_FUNC static inline auto make_ctx() { return PairWeightIteratorCtx{}; }

    inline double get(size_t cell_i, size_t p_i, size_t p_nbh_index,PairWeightIteratorCtx&) const noexcept
    {
      // obsolete, needs to be redefined
      std::abort();
    }
  };


  struct ComputePairNullWeightIterator
  {
    static inline constexpr bool c_has_nbh_data = false;
    static inline constexpr double c_default_value = 1.0;
    struct PairWeightIteratorCtx {};
    ONIKA_HOST_DEVICE_FUNC ONIKA_ALWAYS_INLINE static PairWeightIteratorCtx make_ctx() { return {}; }
    ONIKA_HOST_DEVICE_FUNC ONIKA_ALWAYS_INLINE static double get(size_t,size_t,size_t,PairWeightIteratorCtx&) noexcept { return 1.0; }
  };


  // ---- optional locks ------

  struct FakeParticleLock
  {
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void lock() noexcept {}
    ONIKA_HOST_DEVICE_FUNC static inline constexpr void unlock() noexcept {}
  };

  struct FakeCellParticleLocks
  {
    ONIKA_HOST_DEVICE_FUNC inline FakeParticleLock& operator [] (size_t) const { static FakeParticleLock foo{}; return foo; }
  };

  template<bool UseLocks> struct ComputePairOptionalLocks;
  
  template<> struct ComputePairOptionalLocks<true>
  {
    using CellParticleLocks = spin_mutex_array;
    ONIKA_HOST_DEVICE_FUNC inline CellParticleLocks& operator [] (size_t i) const
    {
      return m_locks[i];
    }
    CellParticleLocks* m_locks = nullptr;
  };
  
  template<> struct ComputePairOptionalLocks<false>
  {
    using CellParticleLocks = FakeCellParticleLocks;
    ONIKA_HOST_DEVICE_FUNC inline CellParticleLocks& operator [] (size_t) const
    {
      static CellParticleLocks foo{};
      return foo;
    }
  };
  
  template<class T> struct CPBufLockReference { using reference_t = T&; };
  template<> struct CPBufLockReference< ComputePairOptionalLocks<false> > { using reference_t = ComputePairOptionalLocks<false>; };
  template<> struct CPBufLockReference<FakeCellParticleLocks> { using reference_t = FakeCellParticleLocks; };
  template<> struct CPBufLockReference<FakeParticleLock> { using reference_t = FakeParticleLock; };
  template<class T> using cpbuf_lock_reference_t = typename CPBufLockReference<T>::reference_t ;
  
  using NullGridParticleLocks = ComputePairOptionalLocks<false>;

  template<class NbhIteratorT,
	   class WeightIteratorT = ComputePairNullWeightIterator, 
	   class TransformT = NullXForm, 
	   class GridParticleLocksT = NullGridParticleLocks , 
	   class CellFilteringFuncT = ComputePairTrivialCellFiltering,
	   class ParticleFilteringFuncT = ComputePairTrivialParticleFiltering >
  static inline
  ComputePairOptionalArgs2<NbhIteratorT,WeightIteratorT,TransformT,GridParticleLocksT, CellFilteringFuncT, ParticleFilteringFuncT > 
  make_compute_pair_optional_args( NbhIteratorT n, WeightIteratorT w={}, TransformT t={}, GridParticleLocksT l={} , CellFilteringFuncT cf={}, ParticleFilteringFuncT pf={} )
  {
    return ComputePairOptionalArgs2<NbhIteratorT,WeightIteratorT,TransformT,GridParticleLocksT,CellFilteringFuncT,ParticleFilteringFuncT> { n, w, t, l, cf , pf };
  }
  
}

