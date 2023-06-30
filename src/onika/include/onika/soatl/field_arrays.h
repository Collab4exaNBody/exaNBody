#pragma once

#include <onika/soatl/packed_field_arrays.h>
#include <onika/soatl/field_tuple.h>
#include <onika/soatl/field_pointer_tuple.h>
#include <onika/soatl/traits.h>
#include <onika/soatl/copy.h>
#include <onika/soatl/field_combiner.h>
#include <onika/memory/alloc_strategy.h>
#include <onika/cuda/cuda.h>
#include <onika/flat_tuple.h>

#include <utility>
#include <type_traits>
		
namespace onika
{

  namespace soatl
  {

    // virtual field Id to get element index in array
/*
    struct _FieldArrayElementIndex {};
    template<> struct FieldId<_FieldArrayElementIndex>
    {
      using value_type = size_t;
      using Id = _FieldArrayElementIndex;
      static const char* name() { return "FieldArrayElementIndex"; }
    };
    using FieldArrayElementIndex = FieldId<_FieldArrayElementIndex>;
    static constexpr FieldArrayElementIndex field_array_index = {};

    struct FieldArrayElementIndexAccessor
    {
      ONIKA_HOST_DEVICE_FUNC inline size_t operator [] (size_t i) const { return i; }
    };
*/

    // field combiner array accessors
    template<class FuncT , class PointerTuple , class FieldIdxSeq> struct FieldArrayCombinerAccessor;
    template<class FuncT , class PointerTuple , std::size_t ... FieldIdx>
    struct FieldArrayCombinerAccessor< FuncT , PointerTuple , std::index_sequence<FieldIdx...> >
    {
      const FuncT m_func;
      const PointerTuple m_pointers;
      using ReturnType = decltype( m_func( ( m_pointers.get(tuple_index<FieldIdx>)[0] ) ... ) );
      ONIKA_HOST_DEVICE_FUNC inline ReturnType operator [] (size_t i) const
      {
        return m_func( m_pointers.get(tuple_index<FieldIdx>)[i] ... );
      }
    };

    /*
     (!) Important note about ALLOCATOR :
     All allocator instances used during the lifetime of a FieldArraysWithAllocator object must be "deallocation compatible" with each other,
     meaning that if different allocators are passed (or default constructed by default argument value) they must be capable
     of deallocating memory allocated by any of the other allocators.
    */
    template<size_t _Alignment, size_t _ChunkSize, typename _DefaultAllocator, size_t _NbStoredPointers, typename... ids >
    struct FieldArraysWithAllocator
    {
      static_assert( _NbStoredPointers <= sizeof...(ids) , "Cannot store more pointers than fields" );
      static_assert( _Alignment > 0 , "Alignment must be strictly positive" );
      static_assert( IsPowerOf2<_Alignment>::value,"alignment must be a power of 2");
      static_assert( _ChunkSize > 0 , "Chunk size must be strictly positive" );

      static constexpr size_t NbStoredPointers = _NbStoredPointers;
      static constexpr size_t AlignmentLog2 = Log2<_Alignment>::value;
      static constexpr size_t Alignment = (1ul<<AlignmentLog2);
      static constexpr size_t AlignmentLowMask = Alignment - 1;
      static constexpr size_t AlignmentHighMask = ~AlignmentLowMask;
      static constexpr size_t ChunkSize = _ChunkSize;
      static constexpr size_t TupleSize = sizeof...(ids);

	    using TupleValueType = FieldTuple<ids...>;
      using FieldIdsTuple = std::tuple< FieldId<ids> ... > ;
      using AllocStrategy = memory::DefaultAllocationStrategy;
      using DefaultAllocator = _DefaultAllocator;
      //using DefaultClearAllocator = typename DefaultAllocator::NullAllocator;

      // alternative to constexpr function has_field, in case compiler as difficulties with constexpr functions
      // to use it, use : typename MyFieldArrays::template HasField < my_field_id >
      template<class field_id> struct HasFieldHelper { static constexpr bool value = find_index_of_id<field_id,ids...>::index != bad_field_index; };
      template<class FuncT , class... fids> struct HasFieldHelper< FieldCombiner<FuncT,fids...> > { static constexpr bool value = ( ... && ( HasFieldHelper<fids>::value ) ); };
      template<class field_id_or_combiner> using HasField = std::integral_constant< bool , HasFieldHelper<field_id_or_combiner>::value >;

      using ArrayTuple = FlatTuple< typename FieldId<ids>::value_type* __restrict__ ... >;
      template<size_t i> using ArrayTupleElement = flat_tuple_element_t<ArrayTuple,i>;
      template<typename _id> using field_value_t = typename FieldId<_id>::value_type;
      template<typename _id> using field_pointer_t = field_value_t<_id> * __restrict__ ;

      template<class _id>
      static inline constexpr bool has_field( FieldId<_id> )
      {
        return HasField<_id>::value;
      }
      template<class FuncT , class... fids>
      static inline constexpr bool has_field( FieldCombiner<FuncT,fids...> )
      {
        return HasField< FieldCombiner<FuncT,fids...> >::value;
      }

      // writes ith tuple
      ONIKA_HOST_DEVICE_FUNC
      inline void set_tuple ( size_t i, const FieldTuple<ids...>& value )
      {
        assert( i < capacity() );
        ( ... , ( (*this)[FieldId<ids>()][i] = value[FieldId<ids>()] ) );
      }
      inline void set_tuple ( size_t i, const typename FieldId<ids>::value_type& ... args )
      {
        set_tuple( i, FieldTuple<ids...>(args...) );
      }
      inline void set_tuple ( size_t i, const std::tuple< typename FieldId<ids>::value_type ... > & value )
      {
        set_tuple( i, FieldTuple<ids...>(value) );
      }

      // read only fields of tuple at position i that exist in this field array. other fields of tuple passed as argument are unchanged.
      template<typename... otherIds>
      ONIKA_HOST_DEVICE_FUNC
      inline void read_tuple( size_t i, FieldTuple<otherIds...>& tp ) const
      {
        assert( i < capacity() );
        ( ... , ( tp[ FieldId<otherIds>() ] = (*this)[FieldId<otherIds>()][i] ) );
      }

      // write only fields of argument tuple that exist in this field array. other fields are unchanged.
      template<typename... otherIds>
      inline void write_tuple( size_t i, const FieldTuple<otherIds...>& tp ) const
      {
        assert( i < capacity() );
        TEMPLATE_LIST_BEGIN
          (*this)[FieldId<otherIds>()][i] = tp[ FieldId<otherIds>() ]
        TEMPLATE_LIST_END
      }

      // reads ith tuple  
      inline FieldTuple<ids...> operator [] ( size_t i ) const
      {
        assert( i < capacity() ); 
        return FieldTuple<ids...>( (*this)[FieldId<ids>()][i] ... );
      }

      inline FieldPointerTuple<Alignment,ChunkSize,ids...> field_pointers() const
      {
        return FieldPointerTuple<Alignment,ChunkSize,ids...>( (*this)[FieldId<ids>()] ... );
      }

      inline void swap( size_t a, size_t b )
      {
        assert( a < size() );
        assert( b < size() );
        auto tmp = (*this)[a];
        set_tuple( a , (*this)[b] );
        set_tuple( b , tmp );
      }

      // adds an element at the end of container, increments size
      template<class Allocator = DefaultAllocator>
      inline void push_back( const FieldTuple<ids...>& value , const Allocator& alloc = Allocator{} )
      {
        const size_t s = size();
        resize( s+1 , alloc );
        set_tuple( s, value );
      }  

      template<class Allocator = DefaultAllocator>
      inline void assign(size_t s , const FieldTuple<ids...>& value , const Allocator& alloc = Allocator{} )
      {
        resize( s , alloc );
        for(size_t i=0;i<s;i++) set_tuple( i, value );
      }

      // resize, and eventually augment capacity
      template<class Allocator = DefaultAllocator>
      inline void resize(size_t s , const Allocator& alloc = Allocator{} )
      {
        if( s != m_size )
        {
          size_t new_capacity = AllocStrategy::update_capacity(s,capacity(),chunksize());
          if( new_capacity != m_capacity )
          {
            reallocate( new_capacity, alloc );
          }
          m_size = s;
        }
      }

      // minimal capacity to hold size() items with respect to constraints (chunk size)
      ONIKA_HOST_DEVICE_FUNC
	    inline size_t minimal_capacity()
	    {
	      return ( (size()+chunksize()-1)/chunksize() ) * chunksize();
      }

      template<class Allocator = DefaultAllocator>
      inline void shrink_to_fit( const Allocator& alloc = Allocator{} , bool force_reallocate=false )
      {
        reallocate( minimal_capacity() , alloc , force_reallocate );
      }

      // utility class method to build a tuple compatible with this container
      static inline FieldTuple<ids...> make_tuple(const typename FieldId<ids>::value_type & ... args)
      {
        return FieldTuple<ids...>( args ... );
      }

      // how many usefull bytes among those allocated
      ONIKA_HOST_DEVICE_FUNC
      inline size_t payload_bytes() const
      {
        size_t n = 0;
		    ( ... , (n += sizeof(typename FieldId<ids>::value_type) ) );
        return n * size();
      }

      // serialize to stream
      template<class FuncT>
      inline void apply_arrays(FuncT f)
      {
        ( ... , ( f( (*this)[FieldId<ids>()] , size() ) ) );
      }

      // get size, chunk ceil and capacity
	    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
	    ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return m_size==0; }
	    ONIKA_HOST_DEVICE_FUNC inline size_t chunk_ceil() const { return ( (size()+chunksize()-1) / chunksize() ) * chunksize(); }
	    ONIKA_HOST_DEVICE_FUNC inline size_t capacity() const { return m_capacity; }
	    
	    // resize to 0 and free all resources.
      template<class Allocator = DefaultAllocator>
      inline void clear( const Allocator& alloc = Allocator{} ) { resize(0,alloc); }

	    static constexpr size_t alignment() { return Alignment; }
	    static constexpr size_t chunksize() { return ChunkSize; }

      inline FieldArraysWithAllocator()
      {
        init();
      }
      
      template<class Allocator = DefaultAllocator>
      inline FieldArraysWithAllocator( size_t n , const Allocator& alloc = Allocator{} )
      {
        init();
        resize( n , alloc );
      }
      /*
      inline FieldArraysWithAllocator(size_t n, const FieldTuple<ids...>& value )
      {
        init();
        resize( n, value );
      }
      inline FieldArraysWithAllocator(size_t n, const typename FieldId<ids>::value_type & ... args )
      {
        init();
        resize( n, FieldTuple<ids...>(args...) );
      }
      inline FieldArraysWithAllocator(size_t n, const std::tuple< typename FieldId<ids>::value_type ... > & value )
      {
        init();
        resize( n, FieldTuple<ids...>(value) );
      }
      */
      inline ~FieldArraysWithAllocator()
	    {
		    assert( empty() );
		    assert( capacity() == 0 );
		    assert( storage_ptr() == nullptr );
	    }

      // **************** field pointer accessors ********************
      template<typename _id>
      ONIKA_HOST_DEVICE_FUNC
      ONIKA_ALWAYS_INLINE 
      field_pointer_t<_id> operator [] ( FieldId<_id> ) const
      {
        return (field_pointer_t<_id>) ONIKA_BUILTIN_ASSUME_ALIGNED(
          field_pointer( FieldId<_id>{}
                       , std::integral_constant<bool,find_index_of_id_v<_id,ids...> < NbStoredPointers >{} // field is stored
                       , std::true_type{} // field exists : required to be true for [] operator
                       ) , Alignment );
      }
/*
      ONIKA_HOST_DEVICE_FUNC
      inline __attribute__((always_inline)) 
      FieldArrayElementIndexAccessor operator [] ( FieldArrayElementIndex ) const
      {
        return {};
      }
*/
      template<typename _id>
      ONIKA_HOST_DEVICE_FUNC
      ONIKA_ALWAYS_INLINE
      field_pointer_t<_id> field_pointer_or_null ( FieldId<_id> ) const
      {
        return (field_pointer_t<_id>) ONIKA_BUILTIN_ASSUME_ALIGNED(
          field_pointer( FieldId<_id>{}
                       , std::integral_constant<bool,find_index_of_id_v<_id,ids...> < NbStoredPointers >{} // field is stored
                       , std::integral_constant<bool,find_index_of_id_v<_id,ids...> < TupleSize >{} // field exists : optional here
                       ) , Alignment );
      }

      // **************** field combiner accessors ********************
      template<class FuncT , class... fids>
      ONIKA_HOST_DEVICE_FUNC
      ONIKA_ALWAYS_INLINE
      auto operator [] ( const FieldCombiner<FuncT,fids...>& combiner ) const
      {
        using PointerTuple = FlatTuple< const decltype(this->operator[](FieldId<fids>{})) ... >;
        return FieldArrayCombinerAccessor<FuncT, PointerTuple , std::make_index_sequence<sizeof...(fids)> > { combiner.m_func , PointerTuple{ this->operator [] (FieldId<fids>{}) ... } };
      }

      //*************** copy/move operators and constructors ***************
      template<class Allocator = DefaultAllocator>
      inline FieldArraysWithAllocator( const FieldArraysWithAllocator & other , const Allocator& alloc = Allocator{} )
      {
        const size_t n = other.size();
        init();
        resize( n , alloc );
        TEMPLATE_LIST_BEGIN
          std::memcpy( (*this)[FieldId<ids>()] , other[FieldId<ids>()] , n * sizeof(typename FieldId<ids>::value_type) )
        TEMPLATE_LIST_END
      }

      inline FieldArraysWithAllocator( FieldArraysWithAllocator && other )
        : m_size( other.m_size )
        , m_capacity( other.m_capacity )
        , m_field_arrays( other.m_field_arrays )
      {
        other.reset();
      }

      template<class Allocator = DefaultAllocator>
      inline FieldArraysWithAllocator& copy_from ( const FieldArraysWithAllocator & other , const Allocator& alloc = Allocator{} )
      {
        const size_t n = other.size();
        m_size = 0; // this keeps the same capacity but avoids copy of elements during next resize
        resize( n , alloc ); // increase capacity if needed
        TEMPLATE_LIST_BEGIN
          std::memcpy( (*this)[FieldId<ids>()] , other[FieldId<ids>()] , n * sizeof(typename FieldId<ids>::value_type) )
        TEMPLATE_LIST_END
        return *this;
      }

      inline FieldArraysWithAllocator& operator = ( FieldArraysWithAllocator && other )
      {
        clear();
        m_size = other.m_size;
        m_capacity = other.m_capacity;
        m_field_arrays = other.m_field_arrays;
        other.reset();
        return *this;
      }

      template<typename... otherIds>
      ONIKA_HOST_DEVICE_FUNC inline void capture_pointers( FieldPointerTuple<Alignment,ChunkSize,otherIds...>& ptrtuple ) const
      {
        ( ... , ( ptrtuple.set_pointer( FieldId<otherIds>() , field_pointer_or_null( FieldId<otherIds>() ) ) ) );
      }

      // access pointer from field index (index in the list of template parameter pack)
      template<size_t i> ArrayTupleElement<i> & pointer_ref_at()
      {
        static_assert(i<NbStoredPointers,"cannot return a reference to non stored pointer");
        return * (ArrayTupleElement<i>*)(m_field_arrays.p+i);
      }

      // access to unique storage pointer
      ONIKA_HOST_DEVICE_FUNC inline void* storage_ptr() const { return m_field_arrays.p[0]; }

      // !!!
      // Dangerous, use only just before calling shrink_to_fit
      // !!!
      template<class Allocator = DefaultAllocator>
      inline void unsafe_storage_free( const Allocator& alloc = Allocator{} )
      {
        assert( capacity() >= size() );
        if( storage_ptr() != nullptr )
        {
          alloc.deallocate( storage_ptr() , capacity() );
          set_storage_ptr( nullptr );
        }
      }

      // !!!
      // Dangerous, use only to generate a view over a raw buffer holding a FieldArrays
      // must call reset() after accessing data. resize() or clear() must never be called before reset()
      // !!!
      inline void unsafe_make_view(void * ptr , size_t sz)
      {
        assert( size() == 0 );
        assert( capacity() == 0 );
        assert( storage_ptr() == nullptr );
        m_size = m_capacity = sz;
        set_storage_ptr( ptr );
      }

      // !!!
      // Dangerous, to be used after unsafe_make_view to restore object to a safe state
      // !!!
      inline void unsafe_reset()
      {
        reset();
      }


      template<class Allocator = DefaultAllocator>
      inline size_t storage_size(const Allocator& alloc = Allocator{}) const
      {
        return alloc.allocation_bytes( capacity() );
      }

      template<class Allocator = DefaultAllocator>
      inline size_t memory_bytes(const Allocator& alloc = Allocator{}) const
      {
        return storage_size(alloc) + sizeof(FieldArraysWithAllocator);
      }

      template<class Allocator = DefaultAllocator>
      inline bool is_gpu_addressable(const Allocator& alloc = Allocator{}) const
      {
        return alloc.is_gpu_addressable( storage_ptr() , capacity() );
      }

    private:

      inline void reset()
      {
        m_size = 0;
        m_capacity = 0;
        init();
      }

      inline void init()
      {
        assert( size() == 0 );
        assert( capacity() == 0 );
        set_storage_ptr( nullptr );
      }

      template<typename _id>
      ONIKA_HOST_DEVICE_FUNC
      ONIKA_ALWAYS_INLINE
      field_pointer_t<_id> field_pointer ( FieldId<_id> , std::true_type , std::true_type ) const
      {    
        return (field_pointer_t<_id>) m_field_arrays.p[ find_index_of_id_v<_id,ids...> ];
      }

      template<typename _id>
      ONIKA_HOST_DEVICE_FUNC
      ONIKA_ALWAYS_INLINE
      field_pointer_t<_id> field_pointer ( FieldId<_id> , std::false_type , std::true_type ) const
      {    
        using preceding_fids = preceding_field_ids_t< _id , ids... >;
        using calculator = pfa_size_calculator_t<Alignment,preceding_fids,ChunkSize>;
        return (field_pointer_t<_id>) ( (uint8_t*)( m_field_arrays.p[0] ) + calculator::storage_size(m_capacity) );
      }

      template<typename _id>
      ONIKA_HOST_DEVICE_FUNC
      static constexpr inline 
      field_pointer_t<_id> field_pointer ( FieldId<_id> , std::false_type , std::false_type )
      {    
        return nullptr;
      }

      // given a pointer to a field and its type, returns the pointer to the next field
      template<class ElementType>
      static inline void pfa_advance_field_ptr( size_t capacity, ElementType* __restrict__ &eptr , void* &ptr )
      {
        eptr = reinterpret_cast<ElementType*>( ptr );
        if( ptr != nullptr )
        {
          ptr = static_cast<uint8_t*>(ptr) + ( ( ( capacity * sizeof(ElementType) ) + AlignmentLowMask ) & AlignmentHighMask );
        }
      }

      template<size_t... StoredPtrIndex>
      inline void _set_storage_ptr( void* ptr, std::integer_sequence<size_t,StoredPtrIndex...> )
      {
		    ( ... , ( pfa_advance_field_ptr( capacity() , pointer_ref_at<StoredPtrIndex>() , ptr ) ) );    
      }
      inline void set_storage_ptr( void* ptr )
      {
        _set_storage_ptr( ptr , std::make_index_sequence<NbStoredPointers>() );
      }

      template<class Allocator = DefaultAllocator>
	    inline void reallocate(size_t s , const Allocator& alloc = Allocator{} , bool force_reallocate = false)
	    {
        assert( capacity() >= size() );
		    assert( ( s % ChunkSize ) == 0 );
		    if( s == capacity() && ! force_reallocate ) return;

        assert( capacity() >= size() );
		    assert( ( s % ChunkSize ) == 0 );

		    if( s == capacity() && ! force_reallocate )
		    { 
		      return;
		    }

        if( s == 0 )
        {
          if( storage_ptr() != nullptr )
          {
            alloc.deallocate( storage_ptr() , capacity() );
            set_storage_ptr( nullptr );
          }
          reset();
          return;
        }

		    // size_t total_space = allocation_size( s );
		    // assert( ( s==0 && total_space==0 ) || ( s!=0 && total_space!=0 ) );

		    void * new_ptr = nullptr;
		    if( /*total_space*/ s > 0 )
		    {
		      new_ptr = alloc.allocate( s /*total_space , alignment()*/ );
		    }

        // in case m_size>0 and m_storage_ptr==nullptr, it means data has been temporarily freed before reallocation and thus copy must be avoided
        bool copy_content = ( storage_ptr() != nullptr );
		    
		    // copy here
		    size_t cs = std::min( s, size_t(m_size) );
		    FieldArraysWithAllocator tmp; //(cs,s,new_ptr);
		    tmp.m_capacity = s;
		    tmp.m_size = cs;
		    assert( tmp.m_size <= tmp.m_capacity );
		    tmp.set_storage_ptr( new_ptr );
		    if( copy_content )
		    {
		      soatl::copy( tmp, *this, 0, cs, FieldId<ids>()... );
		    }
		    if( storage_ptr() != nullptr )
		    {
		      alloc.deallocate( storage_ptr() , capacity() );
		      set_storage_ptr( nullptr );
		    }
		    set_storage_ptr( tmp.storage_ptr() );
		    m_capacity = tmp.capacity();
		    tmp.reset();
		    
		    // check that tmp has been moved, and does not contain anything anymore
		    assert( tmp.storage_ptr() == nullptr );
		    assert( tmp.size() == 0 );
		    assert( tmp.capacity() == 0 );
		    
		    // check that current container holds newly allocated pointer wityh correct capacity
		    assert( storage_ptr() == new_ptr );
		    assert( capacity() == s );
		    assert( storage_ptr()!=nullptr || capacity()==0 );
	    }

	    array_size_t m_size = 0;
	    array_size_t m_capacity = 0;
	    struct { void * /*__restrict__*/ p[NbStoredPointers]; } m_field_arrays; 
    };

    template<size_t A, size_t C, typename... ids > using FieldArrays = FieldArraysWithAllocator< A, C, PackedFieldArraysAllocatorImpl<memory::DefaultAllocator,A,C,ids...> , 1 , ids... >;

    template<typename... ids>
    inline
    FieldArrays<memory::DEFAULT_ALIGNMENT,memory::DEFAULT_CHUNK_SIZE,ids...> make_hybrid_field_arrays(const FieldId<ids>& ...)
    {
	    return FieldArrays<memory::DEFAULT_ALIGNMENT,memory::DEFAULT_CHUNK_SIZE,ids...>();
    }

    template<size_t A, size_t C,typename... ids>
    inline
    FieldArrays<A,C,ids...> make_hybrid_field_arrays(cst::align<A>, cst::chunk<C>, const FieldId<ids>& ...)
    {
	    return FieldArrays<A,C,ids...>();
    }

    // specialization of traits
    template<size_t A, size_t C, typename Al, size_t N, typename... Ids >
    struct IsFieldArrays< FieldArraysWithAllocator<A,C,Al,N,Ids...> >  : public std::true_type {};

  } // namespace soatl
}

