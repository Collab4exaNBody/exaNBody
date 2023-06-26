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

  // forward declaration for friend types
  template<size_t _Alignment, size_t _ChunkSize,typename _Allocator, size_t nsp, typename... ids > struct FieldArraysWithAllocator;


  // ================ Field Arrays Storage Size Calculator =========================
  template<size_t _C> using ChkSz = cst::chunk<_C>;
  template<size_t _A> using AlSz = cst::align<_A>;

  // represents a succession of element types having the same byte size
  template<size_t _ElementSize, size_t _Repeats, class CS, class AL
         , bool UsePadding = ( (_ElementSize*CS::value) % AL::value != 0 ) >
  struct PFASizeCalculatorItem;

  template<size_t _ElementSize, size_t _Repeats, size_t _C , size_t _A>  
  struct PFASizeCalculatorItem<_ElementSize,_Repeats, ChkSz<_C> , AlSz<_A> , true >
  {
    static constexpr bool UsePadding = true;
    static constexpr size_t Alignment = _A;
    static constexpr size_t ChunkSize = _C;
    static constexpr size_t ElementSize = _ElementSize;
    static constexpr size_t Repeats = _Repeats;
    ONIKA_HOST_DEVICE_FUNC static inline size_t storage_size( size_t capacity )
    {
	    static constexpr size_t AlignmentLowMask = Alignment - 1;
	    static constexpr size_t AlignmentHighMask = ~AlignmentLowMask;
      return Repeats * ( ( ( capacity * ElementSize ) + AlignmentLowMask ) & AlignmentHighMask );
    }
  };

  template<size_t _ElementSize, size_t _Repeats, size_t _C , size_t _A>  
  struct PFASizeCalculatorItem<_ElementSize,_Repeats, ChkSz<_C> , AlSz<_A> , false >
  {
    static constexpr bool UsePadding = false;
    static constexpr size_t Alignment = _A;
    static constexpr size_t ChunkSize = _C;
    static constexpr size_t ElementSize = _ElementSize;
    static constexpr size_t Repeats = _Repeats;
    ONIKA_HOST_DEVICE_FUNC static inline size_t storage_size( size_t capacity )
    {
#     ifndef NDEBUG
      using padded_calculator = PFASizeCalculatorItem<ElementSize,Repeats, cst::chunk<ChunkSize> , cst::align<Alignment>,true>;
#     endif
      static_assert( (ElementSize*ChunkSize) % Alignment == 0 , "inconsistent ChunkSize/Alignment for no padding option");
      assert( capacity % ChunkSize == 0 );
      assert( padded_calculator::storage_size(capacity) == capacity * ( Repeats * ElementSize ) );
      return capacity * ( Repeats * ElementSize );
    }
  };

  // array size calculator item merge strategy
  template<class Item1 , class Item2 ,
  // do we need padding
  bool padding = ( (Item1::ElementSize*Item1::ChunkSize) % Item1::Alignment != 0 ) || ( (Item2::ElementSize*Item2::ChunkSize) % Item2::Alignment != 0 ) >
  struct Merge2PFAItems
  {
    static constexpr bool is_valid = ( Item1::ElementSize == Item2::ElementSize );
    static_assert( Item1::ChunkSize == Item2::ChunkSize , "Items chunksize mismatch" );
    static_assert( Item1::Alignment == Item2::Alignment , "Items alignment mismatch" );
    using type = PFASizeCalculatorItem< Item1::ElementSize , Item1::Repeats+Item2::Repeats , ChkSz<Item1::ChunkSize> , AlSz<Item1::Alignment> >;
  };
  // specialization where none of items need padding for the next pointer to be aligned
  // in this case, size computation can be factorized
  template<class Item1 , class Item2>
  struct Merge2PFAItems<Item1,Item2,false>
  {
    static constexpr bool is_valid = true;
    static_assert( Item1::ChunkSize == Item2::ChunkSize , "Items chunksize mismatch" );
    static_assert( Item1::Alignment == Item2::Alignment , "Items alignment mismatch" );
    using type = PFASizeCalculatorItem< Item1::ElementSize*Item1::Repeats+Item2::ElementSize*Item2::Repeats ,1, ChkSz<Item1::ChunkSize> , AlSz<Item1::Alignment> >;
  };
  template<class Item1 , class Item2> using merge_pfa_items_t = typename Merge2PFAItems<Item1,Item2>::type;
  template<class Item1 , class Item2> inline constexpr bool mergeable_pfa_items_v = Merge2PFAItems<Item1,Item2>::is_valid;

  // placeholder for a set of PFASizeCalculatorItem
  template<typename... Items> struct PFASizeCalculatorItems {};

  // Prepend an item to a list of items 
  template<typename Item1, typename Items> struct PFASizeCalculatorItemsPrepend;
  template<typename Item1, typename... Items>
  struct PFASizeCalculatorItemsPrepend< Item1 , PFASizeCalculatorItems<Items...> >
  {
    using type = PFASizeCalculatorItems< Item1, Items... >;
  };
  template<typename Item1, typename Items> using prepend_pfasci_t = typename PFASizeCalculatorItemsPrepend<Item1,Items>::type;

  template<typename Fids, size_t, size_t> struct PFASizeCalculatorItemsFromIdsRaw;
  template<size_t _C,size_t _A, typename... ids> struct PFASizeCalculatorItemsFromIdsRaw< FieldIds<ids...> , _C , _A >
  {
    using type = PFASizeCalculatorItems< PFASizeCalculatorItem<sizeof(typename FieldId<ids>::value_type),1,ChkSz<_C>,AlSz<_A> > ...  >;
  };
  template<typename Fids, size_t _C, size_t _A> using pfascitems_from_fids_raw_t = typename PFASizeCalculatorItemsFromIdsRaw<Fids,_C,_A>::type;

  // compact a list of PFASizeCalculatorItem by merging successive items with the same ElementSize
  template<typename Items> struct PFASizeCalculatorItemsCompacter;
  template<> struct PFASizeCalculatorItemsCompacter< PFASizeCalculatorItems<> >
  {
    using type = PFASizeCalculatorItems<>;
  };
  template<typename Item1> struct PFASizeCalculatorItemsCompacter< PFASizeCalculatorItems<Item1> >
  {
    using type = PFASizeCalculatorItems<Item1>;
  };
  template<typename Item1,typename Item2> struct PFASizeCalculatorItemsCompacter< PFASizeCalculatorItems<Item1,Item2> >
  {
    static_assert( Item1::ChunkSize == Item2::ChunkSize );
    using type = std::conditional_t<
        mergeable_pfa_items_v<Item1,Item2>,
        PFASizeCalculatorItems< merge_pfa_items_t<Item1,Item2> > ,
        PFASizeCalculatorItems<Item1,Item2>
        >;
  };
  template<typename Item1,typename Item2, typename... Items> struct PFASizeCalculatorItemsCompacter< PFASizeCalculatorItems<Item1,Item2,Items...> >
  {
    static_assert( Item1::ChunkSize == Item2::ChunkSize );
    using type = std::conditional_t<
        mergeable_pfa_items_v<Item1,Item2>,
        typename PFASizeCalculatorItemsCompacter< prepend_pfasci_t< merge_pfa_items_t<Item1,Item2> , PFASizeCalculatorItems<Items...> > >::type,
        prepend_pfasci_t< Item1, typename PFASizeCalculatorItemsCompacter< PFASizeCalculatorItems<Item2,Items...> >::type >
        >;
  };

  // finally, we define size calculator items to be the compacted version of the set of items obtained from field ids
  // NPS = no padding size, size abov which padding will never be added after a field array storage space
  template<typename Fids, size_t _C, size_t _A> using pfa_size_calculator_items_from_ids_t = typename PFASizeCalculatorItemsCompacter< pfascitems_from_fids_raw_t<Fids,_C,_A> >::type;

  // final field ids storage size calculator
  template<typename PFAItems> struct PFASizeCalculator;
  template<> struct PFASizeCalculator< PFASizeCalculatorItems<> >
  {
    using PFAItems = PFASizeCalculatorItems<>;
    ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t storage_size(size_t) { return 0; }
  };
  template<typename... Items>
  struct PFASizeCalculator< PFASizeCalculatorItems<Items...> >
  {
    using PFAItems = PFASizeCalculatorItems<Items...>;
    ONIKA_HOST_DEVICE_FUNC static inline size_t storage_size(size_t capacity)
    {
      return ( ... + ( Items::storage_size(capacity) ) );
    }
  };

  template<size_t _A, typename Fids, size_t _C> using pfa_size_calculator_t = PFASizeCalculator< pfa_size_calculator_items_from_ids_t<Fids,_C,_A> >;

  template<size_t _A, size_t _C, typename Fids>
  ONIKA_HOST_DEVICE_FUNC static inline size_t pfa_storage_size(size_t capacity)
  {
    using calculator = pfa_size_calculator_t<_A,Fids,_C>;
    return calculator::storage_size(capacity);
  }

  template<size_t _A, size_t _C, typename id, typename... ids>
  ONIKA_HOST_DEVICE_FUNC static inline size_t pfa_pointer_offset(size_t capacity)
  {
    using preceding_fids = preceding_field_ids_t< id , ids... >;
    using calculator = pfa_size_calculator_t<_A,preceding_fids,_C>;
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

