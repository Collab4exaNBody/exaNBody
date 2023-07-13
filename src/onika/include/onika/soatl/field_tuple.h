#pragma once

#include <cstdlib> // for size_t
#include <type_traits>
#include <cstring>
#include <assert.h>

#include <onika/soatl/constants.h>
#include <onika/variadic_template_utils.h>
#include <onika/flat_tuple.h>
#include <onika/cuda/cuda.h>

namespace onika { namespace soatl
{

  namespace details
  {
    template<typename T, bool = std::is_arithmetic<T>::value, bool = std::is_pointer<T>::value, bool = std::is_default_constructible<T>::value >
    struct ZeroHelper
    {
      static inline void zero(T& x) { std::memset( &x, 0, sizeof(T) ); }
    };
    template<typename T>
    struct ZeroHelper<T,true,false,true>
    {
      ONIKA_HOST_DEVICE_FUNC
      static inline void zero(T& x) { x = 0; }
    };
    template<typename T>
    struct ZeroHelper<T,false,true,true>
    {
      ONIKA_HOST_DEVICE_FUNC
      static inline void zero(T& x) { x = nullptr; }
    };
    template<typename T>
    struct ZeroHelper<T,false,false,true>
    {
      ONIKA_HOST_DEVICE_FUNC
      static inline void zero(T& x) { x = T(); }
    };

    template<typename T>
    ONIKA_HOST_DEVICE_FUNC
    static inline void zero_data(T& x)
    {
      ZeroHelper<T>::zero(x) ;
    }
  }

template<typename... ids>
struct FieldTuple
{
  static constexpr size_t TupleSize = sizeof...(ids);
  using FieldIdsTuple = FlatTuple< FieldId<ids> ... > ;
  using TupleType = FlatTuple< typename FieldId<ids>::value_type ... > ;

  // alternative to constexpr function has_field, in case compiler as difficulties with constexpr functions
  // to use it, use : typename MyFieldArrays::template HasField < my_field_id >
  template<typename field_id>
  using HasField = std::integral_constant< bool, find_index_of_id<field_id,ids...>::index != bad_field_index >;

  template<typename field_id>
  static inline constexpr bool has_field( FieldId<field_id> ) { return HasField<field_id>::value; }

  //! default onstructor
  ONIKA_HOST_DEVICE_FUNC inline FieldTuple() { zero(); }

  //! default copy constructor and copy operator
  /*ONIKA_HOST_DEVICE_FUNC inline */ FieldTuple( const FieldTuple & rhs ) = default; //: m_data(rhs.m_data) { }
  /*ONIKA_HOST_DEVICE_FUNC inline */ FieldTuple& operator = ( const FieldTuple & rhs ) = default; //{ copy_or_zero_fields(rhs); return *this; }

  //! partial copy existing fields in rhs and zero others
  template<typename... other_ids>
  ONIKA_HOST_DEVICE_FUNC
  inline FieldTuple( const FieldTuple<other_ids...> & rhs )
  {
    copy_or_zero_fields( rhs );
  }

  ONIKA_HOST_DEVICE_FUNC
  inline FieldTuple( const typename FieldId<ids>::value_type & ... rhs )
  { 
    ( ... , ( (*this)[ FieldId<ids>() ] = rhs ) );
  }

  // constructor, initialized with a std::tuple
  // inline FieldTuple( const TupleType & rhs ) : m_data(rhs) {}
  // inline TupleType tuple() const { return m_data; }
  
  template<class Func>
  inline void apply_fields( Func f ) const
  {
    ( ... , ( f( FieldId<ids>{} , this->operator [] (FieldId<ids>{}) ) ) );
  }
  
  template<typename _id>
  ONIKA_HOST_DEVICE_FUNC
  inline typename FieldId<_id>::value_type operator [] ( FieldId<_id> ) const
  {
      static constexpr int index = find_index_of_id<_id,ids...>::index;
      return m_data.get(tuple_index<index>);
  }

  template<typename _id>
  ONIKA_HOST_DEVICE_FUNC
  inline typename FieldId<_id>::value_type & operator [] ( FieldId<_id> )
  {
      static constexpr int index = find_index_of_id<_id,ids...>::index;
      return m_data.get(tuple_index<index>);
  }

  template<typename... idsRHS>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_or_zero_fields( const FieldTuple<idsRHS...> & rhs )
  { 
    ( ... , (
      copy_tuple_element_or_zero( rhs.m_data, cst::at<find_index_of_id<ids,ids...>::index>() , cst::at<find_index_of_id<ids,idsRHS...>::index>() )
    ) );
  }

  template<typename... idsRHS>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_existing_fields( const FieldTuple<idsRHS...> & rhs )
  { 
    ( ... , (
      copy_existing_tuple_element( rhs.m_data, cst::at<find_index_of_id<ids,ids...>::index>() , cst::at<find_index_of_id<ids,idsRHS...>::index>() )
    ) );
  }  
  
  ONIKA_HOST_DEVICE_FUNC
  inline void zero()
  {
    ( ... , (
      details::zero_data( (*this)[ FieldId<ids>() ] )
    ) );
  }

  //! partial copy existing fields in rhs and zero others
  ONIKA_HOST_DEVICE_FUNC
  inline bool operator == ( const FieldTuple & rhs ) const
  {
    bool is_equal = true;
    ( ... , (
      is_equal = is_equal && ( (*this)[FieldId<ids>()] == rhs[FieldId<ids>()] )
    ) );
    return is_equal;
  }
  inline bool operator != ( const FieldTuple & rhs ) const { return ! ( *this == rhs ); }
        
//private:
  
  template<size_t index, size_t rhs_index, typename rhs_tuple>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_existing_tuple_element( const rhs_tuple& rhs, cst::at<index>, cst::at<rhs_index> )
  {
      m_data.get(tuple_index<index>) = rhs.get(tuple_index<rhs_index>); //std::get<rhs_index>( rhs );
  }
  
  template<size_t index, typename rhs_tuple>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_existing_tuple_element( const rhs_tuple& rhs, cst::at<index>, cst::at<bad_field_index> ) {}

  template<size_t index, size_t rhs_index, typename rhs_tuple>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_tuple_element_or_zero( const rhs_tuple& rhs, cst::at<index>, cst::at<rhs_index> )
  {
      m_data.get(tuple_index<index>) = rhs.get(tuple_index<rhs_index>); //std::get<rhs_index>( rhs );
  }
  
  template<size_t index, typename rhs_tuple>
  ONIKA_HOST_DEVICE_FUNC
  inline void copy_tuple_element_or_zero( const rhs_tuple& rhs, cst::at<index>, cst::at<bad_field_index> )
  {
    details::zero_data(  m_data.get(tuple_index<index>) );
  }
  
  TupleType m_data;
};

// empty field tuples are forbidden
template<> struct FieldTuple<> {};

template<class Fids> struct _FieldTupleFromFieldIds;
template<class... ids> struct _FieldTupleFromFieldIds< FieldIds<ids...> > { using type = FieldTuple<ids...>; };
template<class Fids> using FieldTupleFromFieldIds = typename _FieldTupleFromFieldIds< Fids >::type ;

template<typename... ids>
inline
FieldTuple<ids...>
make_field_tuple(const FieldId<ids>& ...)
{
	return FieldTuple<ids...>();
}

template<class TupleT,class field_id> static inline constexpr bool field_tuple_has_field_v = TupleT:: template HasField <field_id> ::value;


} // namespace soatl
}

