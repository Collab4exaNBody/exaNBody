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
#include <type_traits>
#include <assert.h>

#include <onika/flat_tuple.h>
#include <onika/soatl/constants.h>
#include <onika/variadic_template_utils.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>

namespace onika { namespace soatl {

template<class... ids>
static inline constexpr FlatTuple< typename FieldId<ids>::value_type * __restrict__ ... > null_field_ptr_tuple_v
  = { static_cast<typename FieldId<ids>::value_type *>(nullptr) ... };

template<size_t _Alignment, size_t _ChunkSize,class... ids>
struct FieldPointerTuple
{
  static_assert( _Alignment > 0 , "Alignment must be strictly positive" );
  static_assert( IsPowerOf2<_Alignment>::value,"alignment must be a power of 2");
  static_assert( _ChunkSize > 0 , "Chunk size must be strictly positive" );
  
  static constexpr size_t AlignmentLog2 = Log2<_Alignment>::value;
  static constexpr size_t Alignment = (1ul<<AlignmentLog2);
  static constexpr size_t ChunkSize = _ChunkSize;
  static constexpr size_t TupleSize = sizeof...(ids);
  using FieldIdsTuple = FlatTuple< FieldId<ids> ... > ;
  using TupleType = FlatTuple< typename FieldId<ids>::value_type * __restrict__ ... > ;

  //! default onstructor
  inline FieldPointerTuple() = default;

  //! copy constructor
  inline FieldPointerTuple( const FieldPointerTuple<Alignment,ChunkSize,ids...> & rhs ) : m_data( rhs.m_data ) {}

  //! move constructor
  inline FieldPointerTuple( FieldPointerTuple<Alignment,ChunkSize,ids...> && rhs ) : m_data( std::move(rhs.m_data) ) {}

  //! partial copy existing fields in rhs and zero others
  template<typename... other_ids>
  inline FieldPointerTuple( const FieldPointerTuple<Alignment,ChunkSize,other_ids...> & rhs )
  {
    copy_or_zero_fields( rhs );
  }

  inline FieldPointerTuple( typename FieldId<ids>::value_type * __restrict__ ... rhs )
  { 
    TEMPLATE_LIST_BEGIN
      set_pointer( FieldId<ids>() , rhs )
    TEMPLATE_LIST_END
  }

  inline FieldPointerTuple( size_t sz, typename FieldId<ids>::value_type * __restrict__ ... rhs )
  : m_size(sz)
  { 
    TEMPLATE_LIST_BEGIN
      set_pointer( FieldId<ids>() , rhs )
    TEMPLATE_LIST_END
  }

  // constructor, initialized with a std::tuple
  inline FieldPointerTuple( const TupleType & rhs ) : m_data(rhs) {}

  inline const TupleType& tuple() const { return m_data; }
  inline TupleType& tuple() { return m_data; }
  
  ONIKA_HOST_DEVICE_FUNC static constexpr inline size_t alignment() { return Alignment; } 
  ONIKA_HOST_DEVICE_FUNC static constexpr inline size_t chunksize() { return ChunkSize; } 

  template<typename _id>
  ONIKA_HOST_DEVICE_FUNC
  inline typename FieldId<_id>::value_type * __restrict__ operator [] ( FieldId<_id> ) const
  {
      using ValueType = typename FieldId<_id>::value_type;
      static constexpr int index = find_index_of_id<_id,ids...>::index;
      static_assert( index != bad_field_index , "bad field id for operator []" );
      return (ValueType* /*__restrict__*/) ONIKA__bultin_assume_aligned( m_data.get(tuple_index<index>) , Alignment );
  }

  template<typename _id>
  ONIKA_HOST_DEVICE_FUNC
  inline void set_pointer( FieldId<_id> , typename FieldId<_id>::value_type * __restrict__ ptr )
  {
      static constexpr int index = find_index_of_id<_id,ids...>::index;
      static_assert( index != bad_field_index , "bad field id in set_pointer" );
      m_data.get(tuple_index<index>) = ptr;
  }

  template<typename... idsRHS>
  inline void copy_or_zero_fields( const FieldPointerTuple<Alignment,ChunkSize,idsRHS...> & rhs )
  { 
    TEMPLATE_LIST_BEGIN
      copy_tuple_element_or_zero( rhs.tuple(), cst::at<find_index_of_id<ids,ids...>::index>() , cst::at<find_index_of_id<ids,idsRHS...>::index>() )
    TEMPLATE_LIST_END      
  }

  template<typename... idsRHS>
  inline void copy_existing_fields( const FieldPointerTuple<Alignment,ChunkSize,idsRHS...> & rhs )
  { 
    TEMPLATE_LIST_BEGIN
      copy_existing_tuple_element( rhs.tuple(), cst::at<find_index_of_id<ids,ids...>::index>() , cst::at<find_index_of_id<ids,idsRHS...>::index>() )
    TEMPLATE_LIST_END      
  }
  
  inline void zero()
  {
    TEMPLATE_LIST_BEGIN
      set_pointer( FieldId<ids>() , nullptr )
    TEMPLATE_LIST_END
  }

  inline bool operator == ( const FieldPointerTuple& tp ) const
  {
    return m_data == tp.m_data ;
  }

  inline FieldPointerTuple& operator = ( const FieldPointerTuple& tp ) = default;

private:
  
  template<size_t index, size_t rhs_index, typename rhs_tuple>
  inline void copy_existing_tuple_element( const rhs_tuple& rhs, cst::at<index>, cst::at<rhs_index> )
  {
      m_data.get(tuple_index<index>) = rhs.get(tuple_index<rhs_index>);
  }
  
  template<size_t index, typename rhs_tuple>
  inline void copy_existing_tuple_element( const rhs_tuple& rhs, cst::at<index>, cst::at<bad_field_index> ) {}

  template<size_t index, size_t rhs_index, typename rhs_tuple>
  inline void copy_tuple_element_or_zero( const rhs_tuple& rhs, cst::at<index>, cst::at<rhs_index> )
  {
      m_data.get(tuple_index<index>) = rhs.get(tuple_index<rhs_index>);
  }
  
  template<size_t index, typename rhs_tuple>
  inline void copy_tuple_element_or_zero( const rhs_tuple& rhs, cst::at<index>, cst::at<bad_field_index> )
  {
      m_data.get(tuple_index<index>) = nullptr;
  }
  
  inline size_t size() const { return m_size; }

  TupleType m_data = null_field_ptr_tuple_v<ids...>;
  const size_t m_size = 0;
};

// empty field tuples are forbidden
template<size_t _Alignment, size_t _ChunkSize> struct FieldPointerTuple<_Alignment,_ChunkSize> {};

template<size_t A, size_t C, typename... ids>
FieldPointerTuple<A,C,ids...> make_field_pointer_tuple( cst::align<A>, cst::chunk<C>, FieldId<ids>... )
{
  return FieldPointerTuple<A,C,ids...>();
}

template<typename ArrayT, typename... ids>
FieldPointerTuple<ArrayT::Alignment,ArrayT::ChunkSize,ids...> make_field_arrays_view( const ArrayT& a, FieldId<ids>... )
{
  return FieldPointerTuple<ArrayT::Alignment,ArrayT::ChunkSize,ids...>( a.size() , a[FieldId<ids>()] ... );
}


} // namespace soatl

} // namespace onika


