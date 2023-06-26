#pragma once

#include <onika/soatl/field_id.h>
#include <onika/variadic_template_utils.h>
#include <onika/memory/simd.h>

// TODO: OpenMP parallel version for apply. with handling of alignment and chunksizes

namespace onika { namespace soatl
{

// Non-SIMD versions

template<typename OperatorT, typename... T>
static inline void apply( OperatorT f, size_t N, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_pointers_aliasing( N , arraypack ... );
#	endif

	for(size_t i=0;i<N;i++)
	{
		f( arraypack[i] ... );
	}
}

template<typename OperatorT, typename... T>
static inline void apply( OperatorT f, size_t first, size_t N, T* __restrict__ ... arraypack )
{
	apply( f, N, arraypack+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply( OperatorT f, size_t first, size_t N, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	apply( f, N, arrays[fids]+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply( OperatorT f, size_t N, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	apply( f, N, arrays[fids] ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply( OperatorT f, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	apply( f, arrays.size(), arrays[fids] ... );
}

// Non-SIMD parallel versions

template<typename OperatorT, typename... T>
static inline void parallel_apply( OperatorT f, size_t N, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_pointers_aliasing( N , arraypack ... );
#	endif

# pragma omp parallel for
	for(size_t i=0;i<N;i++)
	{
		f( arraypack[i] ... );
	}
}

template<typename OperatorT, typename... T>
static inline void parallel_apply( OperatorT f, size_t first, size_t N, T* __restrict__ ... arraypack )
{
	parallel_apply( f, N, arraypack+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply( OperatorT f, size_t first, size_t N, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	parallel_apply( f, N, arrays[fids]+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply( OperatorT f, size_t N, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	parallel_apply( f, N, arrays[fids] ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply( OperatorT f, FieldArraysT& arrays, const FieldId<ids> & ... fids )
{
	parallel_apply( f, arrays.size(), arrays[fids] ... );
}



// ***** SIMD versions *****


// raw pointers
template<typename OperatorT, typename... T>
static inline void apply_simd( OperatorT f, size_t N, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_simd_pointers( N , arraypack ... );
#	endif

#	pragma omp simd
	for(size_t i=0;i<N;i++)
	{
		f( arraypack[i] ... );
	}
}

template<typename OperatorT, size_t VECSIZE, typename... T>
static inline void apply_simd( OperatorT f, size_t N, cst::chunk<VECSIZE>, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_simd_pointers( N , arraypack ... );
#	endif

	for(size_t i=0;i<N;i+=VECSIZE)
	{
#		pragma omp simd
		for(size_t j=0;j<VECSIZE;j++)
		{
			f( arraypack[i+j] ... );
		}
	}
}

template<typename OperatorT, typename... T>
static inline void apply_simd( OperatorT f, size_t N, cst::chunk<1>, T* __restrict__ ... arraypack )
{
	apply_simd(f,N,arraypack...);
}

template<typename OperatorT, typename... T>
static inline void apply_simd( OperatorT f, size_t first, size_t N, T* __restrict__ ... arraypack )
{
	apply_simd( f, N, arraypack+first ... );
}


// field arrays

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply_simd( OperatorT f, size_t first, size_t N, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
	// in this case chunk size  cannot be guaranted anymore (unless we check first value at runtime, which compiler will do better on its own)
	apply_simd( f, N, arrays[fids]+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply_simd( OperatorT f, size_t N, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
#	ifndef NDEBUG
	TEMPLATE_LIST_BEGIN
		assert( ( arrays.alignment() % memory::SimdRequirements< typename soatl::FieldId<ids>::value_type >::alignment ) == 0 ) 
	TEMPLATE_LIST_END
#	endif

	apply_simd( f, N, cst::chunk<FieldArraysT::ChunkSize>(), arrays[fids] ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void apply_simd( OperatorT f, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
#	ifndef NDEBUG
	TEMPLATE_LIST_BEGIN
		assert( ( arrays.alignment() % memory::SimdRequirements< typename soatl::FieldId<ids>::value_type >::alignment ) == 0 ) 
	TEMPLATE_LIST_END
#	endif

	apply_simd( f, arrays.size(), cst::chunk<FieldArraysT::ChunkSize>(), arrays[fids] ... );
}





// ***** parallel SIMD versions *****


// raw pointers
template<typename OperatorT, typename... T>
static inline void parallel_apply_simd( OperatorT f, size_t N, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_simd_pointers( N , arraypack ... );
#	endif

#	pragma omp parallel for simd
	for(size_t i=0;i<N;i++)
	{
		f( arraypack[i] ... );
	}
}

template<typename OperatorT, size_t VECSIZE, typename... T>
static inline void parallel_apply_simd( OperatorT f, size_t N, cst::chunk<VECSIZE>, T* __restrict__ ... arraypack )
{
#	ifndef NDEBUG
	memory::check_simd_pointers( N , arraypack ... );
#	endif

  N=(N+VECSIZE-1)/VECSIZE;
  
# pragma omp parallel for
	for(size_t i=0;i<N;i++)
	{
#		pragma omp simd
		for(size_t j=0;j<VECSIZE;j++)
		{
			f( arraypack[i*VECSIZE+j] ... );
		}
	}
}

template<typename OperatorT, typename... T>
static inline void parallel_apply_simd( OperatorT f, size_t N, cst::chunk<1>, T* __restrict__ ... arraypack )
{
	parallel_apply_simd(f,N,arraypack...);
}

template<typename OperatorT, typename... T>
static inline void parallel_apply_simd( OperatorT f, size_t first, size_t N, T* __restrict__ ... arraypack )
{
	parallel_apply_simd( f, N, arraypack+first ... );
}


// field arrays

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply_simd( OperatorT f, size_t first, size_t N, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
	// in this case chunk size  cannot be guaranted anymore (unless we check first value at runtime, which compiler will do better on its own)
	parallel_apply_simd( f, N, arrays[fids]+first ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply_simd( OperatorT f, size_t N, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
#	ifndef NDEBUG
	TEMPLATE_LIST_BEGIN
		assert( ( arrays.alignment() % memory::SimdRequirements< typename soatl::FieldId<ids>::value_type >::alignment ) == 0 ) 
	TEMPLATE_LIST_END
#	endif

	parallel_apply_simd( f, N, cst::chunk<FieldArraysT::ChunkSize>(), arrays[fids] ... );
}

template<typename OperatorT, typename FieldArraysT, typename... ids>
static inline void parallel_apply_simd( OperatorT f, FieldArraysT& arrays, const FieldId<ids>& ... fids )
{
#	ifndef NDEBUG
	TEMPLATE_LIST_BEGIN
		assert( ( arrays.alignment() % memory::SimdRequirements< typename soatl::FieldId<ids>::value_type >::alignment ) == 0 ) 
	TEMPLATE_LIST_END
#	endif

	parallel_apply_simd( f, arrays.size(), cst::chunk<FieldArraysT::ChunkSize>(), arrays[fids] ... );
}

} } // namespace soatl


