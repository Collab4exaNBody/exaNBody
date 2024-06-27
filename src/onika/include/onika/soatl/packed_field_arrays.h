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

#include <cstdlib> // for size_t
#include <cassert>

#include <onika/cuda/cuda.h>

#include <onika/soatl/constants.h>
#include <onika/soatl/stdtypes.h>
#include <onika/soatl/traits.h>

namespace onika
{

  namespace soatl
  {

    namespace pfa_details
    {
    
      template<size_t _SZ, size_t _N> struct SizeCount
      {
        static constexpr size_t SIZE = _SZ;
        static constexpr size_t N = _N;
      };

      template<size_t ALIGNMENT, size_t CHUNK_SIZE, class... SzCnts> struct SizeCounts
      {
        ONIKA_HOST_DEVICE_FUNC
        ONIKA_ALWAYS_INLINE
        static size_t storage_size(size_t capacity)
        {
          constexpr size_t CA = ALIGNMENT / CHUNK_SIZE ;
          constexpr size_t CA_lowmask = CA - 1;
          constexpr size_t CA_himask = ~ CA_lowmask;
          capacity = capacity / CHUNK_SIZE;
          return CHUNK_SIZE * ( ... + (
		           		SzCnts::N * ( ( capacity * SzCnts::SIZE + CA - 1 ) & CA_himask )
		                      ));
        }
        template<class StreamT> static inline StreamT& print(StreamT & out )
        {
          out<<"SizeCounts<"<<ALIGNMENT<<","<<CHUNK_SIZE;
          ( ... , ( out<<" , (sz="<<SzCnts::SIZE<<",n="<<SzCnts::N<<")" ) );
          out << " >\n";
          return out;
        }
      };

      template<size_t ALIGNMENT, size_t CHUNK_SIZE>
      struct SizeCounts< ALIGNMENT, CHUNK_SIZE >
      {
        ONIKA_HOST_DEVICE_FUNC
        ONIKA_ALWAYS_INLINE
        static constexpr size_t storage_size(size_t capacity) { return 0; }
        template<class StreamT>
        static inline StreamT& print(StreamT & out )
        {
          out<<"SizeCounts<"<<ALIGNMENT<<","<<CHUNK_SIZE<<",empty-set>"<< "\n";
          return out;
        }
      };

      template<size_t ALIGNMENT, size_t CHUNK_SIZE, size_t NCA, class... SzCnts>
      struct SizeCounts< ALIGNMENT, CHUNK_SIZE, SizeCount<ALIGNMENT/CHUNK_SIZE,NCA> , SzCnts... >
      {
        ONIKA_HOST_DEVICE_FUNC
        ONIKA_ALWAYS_INLINE
        static size_t storage_size(size_t capacity)
        {
          constexpr size_t CA = ALIGNMENT / CHUNK_SIZE ;
          constexpr size_t CA_himask = ~ (CA - 1);
          capacity = capacity / CHUNK_SIZE;
          if constexpr ( sizeof...(SzCnts) > 0 ) return CHUNK_SIZE * ( capacity * CA * NCA + ( ... + ( SzCnts::N * ( ( capacity * SzCnts::SIZE + CA - 1 ) & CA_himask ) )) );
          return CHUNK_SIZE * capacity * CA * NCA;
        }
        template<class StreamT>
        static inline StreamT& print(StreamT & out )
        {
          out<<"SizeCounts<"<<ALIGNMENT<<","<<CHUNK_SIZE<<" , (sz="<< ALIGNMENT / CHUNK_SIZE <<",n="<<NCA<<")*";
          ( ... , ( out<<" , (sz="<<SzCnts::SIZE<<",n="<<SzCnts::N<<")" ) );
          out << " >\n";
          return out;
        }
      };

      template<size_t A, size_t C, class... ids>
      ONIKA_HOST_DEVICE_FUNC
      static inline size_t check_size_for_capacity( std::integral_constant<size_t,A> , std::integral_constant<size_t,C> , FieldIds<ids...> , size_t capacity )
      {
        static_assert( A >= 1 );
        assert( capacity % C == 0 );
        size_t N=0, S=0, P=0;
        ( ... , (
        	S = capacity * sizeof( typename FieldId<ids>::value_type ) ,
	        P = ( A - ( S % A ) ) % A ,
	        N += S + P
        ));
        return N;
      }

      template<size_t A, size_t C>
      struct PFACalc
      {
        static_assert( A % C == 0 );
        static constexpr size_t MAX_FIELD_SIZE = 1 + (A/C);
        size_t size[MAX_FIELD_SIZE] = {};
        size_t count[MAX_FIELD_SIZE] = {};
        size_t n_sizes = 0;
        template<class... Args> inline constexpr PFACalc( std::tuple<Args ...> )
        {
          constexpr size_t CA = A / C;
          for(size_t i=0;i<MAX_FIELD_SIZE;i++) { size[i] = i; count[i] = 0; }
          ( ... , (
	          count[sizeof(Args)%CA] += 1 ,
	          count[CA] += sizeof(Args)/CA
	          ) );
          count[0] = count[CA];
          size[0] = size[CA];
          count[CA] = 0;
          for(size_t i=0;i<MAX_FIELD_SIZE;i++) if( count[i] != 0 ) { count[n_sizes]=count[i]; size[n_sizes]=size[i]; ++n_sizes; }
        }
      };

      template< size_t A, size_t C, class ArgsTupleT , size_t... I>
      static constexpr auto make_size_counts( std::integral_constant<size_t,A> , std::integral_constant<size_t,C> , ArgsTupleT , std::index_sequence<I...> )
      {
        constexpr PFACalc<A,C> pfa( ArgsTupleT{} );
        using SizeCountsT = SizeCounts< A, C, SizeCount< pfa.size[I], pfa.count[I] > ... >;
        return SizeCountsT{};
      };

      template< size_t A, size_t C, class Fids> struct MakeSizeCounts;
      template< size_t A, size_t C, class... ids> struct MakeSizeCounts<A,C,FieldIds<ids...> >
      {
        using ArgsT = std::tuple< typename FieldId<ids>::value_type ... >;
        static constexpr size_t n_sizes = PFACalc<A,C>( ArgsT{} ).n_sizes;
        using type = decltype( make_size_counts( std::integral_constant<size_t,A>{} , std::integral_constant<size_t,C>{} , ArgsT{} , std::make_index_sequence<n_sizes>{} ) );
      };

    } // end of pfa_details

    template<size_t _A, typename Fids, size_t _C> using pfa_size_calculator_t = typename pfa_details::MakeSizeCounts< _A, _C, Fids >::type;

    template<size_t _A, size_t _C, typename Fids>
    ONIKA_HOST_DEVICE_FUNC static inline size_t pfa_storage_size(size_t capacity)
    {
      using calculator = pfa_size_calculator_t<_A,Fids,_C>;
      assert( calculator::storage_size(capacity) == pfa_details::check_size_for_capacity( std::integral_constant<size_t,_A>{} , std::integral_constant<size_t,_C>{} , Fids{} , capacity ) );
      return calculator::storage_size(capacity);
    }

    template<size_t _A, size_t _C, typename id, typename... ids>
    ONIKA_HOST_DEVICE_FUNC static inline size_t pfa_pointer_offset(size_t capacity)
    {
      using preceding_fids = preceding_field_ids_t< id , ids... >;
      using calculator = pfa_size_calculator_t<_A,preceding_fids,_C>;
      assert( calculator::storage_size(capacity) == pfa_details::check_size_for_capacity( std::integral_constant<size_t,_A>{} , std::integral_constant<size_t,_C>{} , preceding_fids{} , capacity ) );
      return calculator::storage_size(capacity);
    }




    /**************************************
     *** Packed field arrays allocators ***
     **************************************/

    class PackedFieldArraysAllocator
    {
    public:
      virtual size_t allocation_bytes(size_t n_elements) const =0;
      virtual void* allocate(size_t n_elements) const =0;
      virtual void deallocate(void* ptr, size_t n_elements) const =0;
      virtual bool is_gpu_addressable(void* ptr, size_t n_elements) const =0;
      virtual bool allocates_gpu_addressable() const =0;
      virtual void set_gpu_addressable_allocation(bool) =0;
    };

    template<class BaseAllocT, size_t Alignment, size_t ChunkSize, class... ids>
    class PackedFieldArraysAllocatorImpl : public PackedFieldArraysAllocator
    {
    public:
      // allocator that doesn't allocate anything. it's usefull to free pointers about which we don't known allocated size
      // BaseAllocT allocator si supposed to have specific behavior when pointer is not null and size is 0 (free without size information)
  //    using NullAllocator = PackedFieldArraysAllocatorImpl<BaseAllocT,Alignment,ChunkSize>;
      
      PackedFieldArraysAllocatorImpl() = default;
      virtual ~PackedFieldArraysAllocatorImpl() = default;

      inline PackedFieldArraysAllocatorImpl(const BaseAllocT& base_alloc) : m_alloc( base_alloc ) {}
      
      inline size_t allocation_bytes(size_t n_elements) const override final
      {
        return pfa_storage_size<Alignment,ChunkSize,FieldIds<ids...> >( n_elements );
      }
      inline void* allocate(size_t n_elements) const override final
      {
        return m_alloc.allocate( allocation_bytes(n_elements) , Alignment );
      }
      inline void deallocate(void* ptr, size_t n_elements) const override final
      {
        m_alloc.deallocate( ptr , allocation_bytes(n_elements) );
      }
      inline bool is_gpu_addressable(void* ptr, size_t n_elements) const override final
      {
        return m_alloc.is_gpu_addressable( ptr , allocation_bytes(n_elements) );
      }
      inline bool allocates_gpu_addressable() const override final
      {
        return m_alloc.allocates_gpu_addressable();
      }
      inline void set_gpu_addressable_allocation(bool yn) override final
      {
        m_alloc.set_gpu_addressable_allocation( yn );
      }

      inline BaseAllocT& base_allocator() { return m_alloc; }
    private:
      BaseAllocT m_alloc;
    };

    template<class BaseAllocT, size_t Alignment, size_t ChunkSize, class Fids>
    struct PackedFieldArraysAllocatorImplFromFieldIds;
    
    template<class BaseAllocT, size_t Alignment, size_t ChunkSize, class... ids>
    struct PackedFieldArraysAllocatorImplFromFieldIds<BaseAllocT,Alignment,ChunkSize, FieldIds<ids...> >
    {
      using type = PackedFieldArraysAllocatorImpl<BaseAllocT,Alignment,ChunkSize,ids...>;
    };
    template<class BaseAllocT, size_t Alignment, size_t ChunkSize, class Fids>
    using pfa_allocator_impl_from_field_ids_t = typename PackedFieldArraysAllocatorImplFromFieldIds<BaseAllocT,Alignment,ChunkSize,Fids>::type;


  } // namespace soatl

} // namespace onika

