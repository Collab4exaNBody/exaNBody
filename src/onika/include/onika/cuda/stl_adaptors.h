#pragma once

#include <vector>
#include <onika/cuda/cuda.h>
#include <onika/cuda/ro_shallow_copy.h>

namespace onika
{

  namespace cuda
  {

    template<class T, class U>
    struct pair
    {
      T first;
      U second;
      ONIKA_HOST_DEVICE_FUNC inline bool operator != ( const pair& other ) const { return first!=other.first || second!=other.second; }
      ONIKA_HOST_DEVICE_FUNC inline bool operator == ( const pair& other ) const { return first==other.first || second==other.second; }
      ONIKA_HOST_DEVICE_FUNC inline bool operator < ( const pair& other ) const { return first<other.first || ( first==other.first && second<other.second ); }
      ONIKA_HOST_DEVICE_FUNC inline bool operator >= ( const pair& other ) const { return first>other.first || ( first==other.first && second>=other.second ); }
    };

    ONIKA_HOST_DEVICE_FUNC inline void memmove( void* dst , const void* src , unsigned int n )
    {
      static_assert( sizeof(uint64_t) == 8 && sizeof(uint8_t) == 1 , "Wrong standard type sizes" );
      if( dst < src )
      {
        volatile uint8_t* dst1 = (uint8_t*) dst;
        const volatile uint8_t* src1 = (const uint8_t*) src;

        volatile uint64_t* dst8 = (uint64_t*) dst;
        const volatile uint64_t* src8 = (const uint64_t*) src;
        unsigned int n8 = n / 8;
        
        unsigned int i;
        for(i=0;i<n8;i++) dst8[i] = src8[i];
        i *= 8;
        for(;i<n;i++) dst1[i] = src1[i];
      }
      else if( dst > src )
      {
        // not implemented yet
        ONIKA_CU_ABORT();
      }
      // else { /* nothing to do */ }
    }

    template<class T, class A>
    struct CudaStdVectorAccess : public std::vector<T,A>
    {
      ONIKA_HOST_DEVICE_FUNC inline const T* _cudata() const { return this->std::vector<T,A>::_M_impl._M_start; }
      ONIKA_HOST_DEVICE_FUNC inline T* _cudata() { return this->std::vector<T,A>::_M_impl._M_start; }
      ONIKA_HOST_DEVICE_FUNC inline size_t _cusize() const
      {
        return size_t( this->std::vector<T,A>::_M_impl._M_finish - this->std::vector<T,A>::_M_impl._M_start );
      }
    };
      
    template<class T, class A>
    ONIKA_HOST_DEVICE_FUNC inline T* vector_data( std::vector<T,A>& v )
    {
      static_assert( sizeof(CudaStdVectorAccess<T,A>) == sizeof(std::vector<T,A>) && alignof(CudaStdVectorAccess<T,A>) == alignof(std::vector<T,A>) );
      CudaStdVectorAccess<T,A>* va = reinterpret_cast<CudaStdVectorAccess<T,A>*>( &v );
      return va->_cudata();
    }
    
    template<class T, class A>
    ONIKA_HOST_DEVICE_FUNC inline const T* vector_data( const std::vector<T,A>& v )
    {
      static_assert( sizeof(CudaStdVectorAccess<T,A>) == sizeof(std::vector<T,A>) && alignof(CudaStdVectorAccess<T,A>) == alignof(std::vector<T,A>) );
      const CudaStdVectorAccess<T,A>* va = reinterpret_cast<const CudaStdVectorAccess<T,A>*>( &v );
      return va->_cudata();
    }

    template<class T, class A>
    ONIKA_HOST_DEVICE_FUNC inline size_t vector_size( const std::vector<T,A>& v )
    {
      static_assert( sizeof(CudaStdVectorAccess<T,A>) == sizeof(std::vector<T,A>) && alignof(CudaStdVectorAccess<T,A>) == alignof(std::vector<T,A>) );
      const CudaStdVectorAccess<T,A>* va = reinterpret_cast<const CudaStdVectorAccess<T,A>*>( &v );
      return va->_cusize();
    }

    template<class T> ONIKA_HOST_DEVICE_FUNC inline size_t vector_size( const VectorShallowCopy<T>& v ) { return v.size(); }
    template<class T> ONIKA_HOST_DEVICE_FUNC inline const T* vector_data( const VectorShallowCopy<T>& v ) { return v.data(); }
    template<class T> ONIKA_HOST_DEVICE_FUNC inline T* vector_data( VectorShallowCopy<T>& v ) { return v.data(); }

    template<class Iterator, class T>
    ONIKA_HOST_DEVICE_FUNC inline Iterator lower_bound( Iterator begin , Iterator end , const T& x )
    {
      while( begin != end )
      {
        if( ! ( (*begin) < x ) ) return begin;
        ++begin;
      }
      return end;
    }
 
  }

}

