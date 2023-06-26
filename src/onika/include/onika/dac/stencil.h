#pragma once

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <onika/oarray.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <functional>

#include <onika/dac/constants.h>
#include <onika/dac/slices.h>

#include <onika/oarray_stream.h>

  // =========== data access control constants ===========
namespace onika
{
  namespace dac
  {

    // **************** access stencil ***********************

    template<class RoSlices, class RwSlices, int... RelPos> struct stencil_element_t
    {
      using ro_slices_t = slices_to_slice_accesses_t<ro_t,RoSlices>;
      using rw_slices_t = slices_to_slice_accesses_t<rw_t,RwSlices>;
      using ro_rw_slices_t = concat_slice_accesses_t<ro_slices_t,rw_slices_t>;
      static inline constexpr size_t Nd = sizeof...(RelPos);
      static inline constexpr bool is_rw_v = ( RwSlices::nb_slices > 0 );
      static inline constexpr oarray_t<int,Nd> relpos_v = { RelPos... };
    };
    //template<class RoSlices, class RwSlices,class Is> struct StencilCenterHelper;
    //template<class RoSlices, class RwSlices,int... Is> struct StencilCenterHelper<RoSlices,RwSlices,std::integer_sequence<int,Is...> > { using type = stencil_element_t<RoSlices,RwSlices,(Is*0)...>; };
    //template<class RoSlices, class RwSlices,int Nd> using stencil_center_t = typename StencilCenterHelper<RoSlices,RwSlices,std::make_integer_sequence<int,Nd> >::type;

    template<class T> struct is_stencil_element_t : std::false_type {};
    template<class T, class U,int... I> struct is_stencil_element_t< stencil_element_t<T,U,I...> > : std::true_type {};
    template<class... T> static inline constexpr bool is_stencil_element_v = ( ... && (is_stencil_element_t<T>::value) );

    template<class... T> struct stencil_elements_t {};
    template<class T, class... U> struct stencil_elements_t<T,U...> { using first_element_t = T; };

    template<class C, class Es, size_t _Scale=1> struct Stencil ;
    template<class C, size_t _Scale, class... E> struct Stencil<C,stencil_elements_t<E...>,_Scale>
    {
      using central_t = C;
      using neighbors_t = stencil_elements_t<E...>;
      static_assert( is_stencil_element_v<C,E...> && C::Nd==0 , "not a conformant access stencil" );
      static inline constexpr unsigned int Scale = _Scale;
      static inline constexpr bool is_rw_v = ( C::is_rw_v || ... || (E::is_rw_v) );
      static inline constexpr unsigned int scaling() { return Scale; }
      static inline constexpr unsigned int ndims()
      {
        if constexpr ( sizeof...(E) >= 1 ) return neighbors_t::first_element_t::Nd;
        return 0;
      }
      static inline constexpr oarray_t<int,ndims()> low_corner()
      {
        oarray_t<int,ndims()> r = ZeroArray<int,ndims()>::zero;
        ( ... , ( r = array_min(r,E::relpos_v) ) );
        return r;
      }
      static inline constexpr oarray_t<int,ndims()> high_corner()
      {
        oarray_t<int,ndims()> r = ZeroArray<int,ndims()>::zero;
        ( ... , ( r = array_max(r,E::relpos_v) ) );
        return r;
      }
    };

    template<class C, class... E> using stencil_t = Stencil<C,stencil_elements_t<E...> >;
    template<size_t S, class C, class... E> using scaled_stencil_t = Stencil<C,stencil_elements_t<E...> , S >;
    template<class ROSlice, class RWSlice> using local_stencil_t = Stencil< stencil_element_t<ROSlice,RWSlice> , stencil_elements_t<> >;
    template<class... Slices> using local_ro_stencil_t = Stencil< stencil_element_t< DataSlices<Slices...> , DataSlices<> > , stencil_elements_t<> >;
    template<class... Slices> using local_rw_stencil_t = Stencil< stencil_element_t< DataSlices<> , DataSlices<Slices...> > , stencil_elements_t<> >;

    template<size_t S, size_t F, class C, class... E>
    static inline constexpr Stencil<C,stencil_elements_t<E...>,S*F> downscale_stencil( Stencil<C,stencil_elements_t<E...>,S> , std::integral_constant<size_t,F> ) { return {}; }

    template<class CenterElement, class NbhRoSlices, class NbhRwSlices, size_t Scale=1>
    struct Nbh3DStencil
    {
      using type = Stencil< CenterElement , stencil_elements_t<
            stencil_element_t<NbhRoSlices,NbhRwSlices,-1,-1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0,-1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1,-1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 0,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 0,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 0,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 1,-1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1,-1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0,-1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1,-1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 0, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 0, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 0, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 1, 0>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1,-1, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0,-1, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1,-1, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 0, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 0, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 0, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices,-1, 1, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 0, 1, 1>
          , stencil_element_t<NbhRoSlices,NbhRwSlices, 1, 1, 1> 
          > , Scale
          >;
    };
    template<class CenterElement, size_t Scale>
    struct Nbh3DStencil<CenterElement, DataSlices<>, DataSlices<>, Scale >
    {
      using type = Stencil< CenterElement , stencil_elements_t<> , Scale >;
    };
    template<class CenterElement, class NbhRoSlices, class NbhRwSlices, size_t Scale=1 > using nbh_3d_stencil_t = typename Nbh3DStencil<CenterElement,NbhRoSlices,NbhRwSlices,Scale>::type;

    template<class S> struct IsEmptyStencilElement : public std::false_type {};
    template<> struct IsEmptyStencilElement< stencil_element_t< DataSlices<> , DataSlices<> > > : public std::true_type {};
    template<class S> static inline constexpr bool is_empty_stencil_element_v = IsEmptyStencilElement<S>::value;
    
    template<class S> struct StencilAllNbhAreEmpty : public std::false_type {};
    template<class C, size_t S, class ... NSE> struct StencilAllNbhAreEmpty< Stencil<C,stencil_elements_t<NSE...>,S>  >
    {
      static inline constexpr bool value = ( ... && ( is_empty_stencil_element_v<NSE> ) );
    };
    template<class S> static inline constexpr bool has_empty_neighborhood_v = StencilAllNbhAreEmpty<S>::value;

    template<class S, bool = has_empty_neighborhood_v<S> > struct RemoveEmptyNeighborhood;
    template<class S> struct RemoveEmptyNeighborhood<S,false> { using type = S; };
    template<class S> struct RemoveEmptyNeighborhood<S,true> { using type = Stencil<typename S::central_t,stencil_elements_t<>,S::Scale>; };
    template<class S> using remove_empty_neighborhood_t = typename RemoveEmptyNeighborhood<S>::type;

    template<class S> struct IsLocalStencil : public std::false_type {};
    template<class C, size_t S> struct IsLocalStencil< Stencil<C,stencil_elements_t<>,S> > : std::true_type {};
    template<class S> static inline constexpr bool is_local_stencil_v = IsLocalStencil<S>::value;


    struct AbstractStencil
    {
      static inline constexpr size_t MAX_BIT_STORAGE_WORDS = 128;
      static inline constexpr size_t MAX_BIT_STORAGE = 64 * MAX_BIT_STORAGE_WORDS;
      static inline constexpr int MAX_DIMS = 4;
      uint64_t m_data[MAX_BIT_STORAGE_WORDS];
      int8_t m_low[MAX_DIMS];
      uint8_t m_size[MAX_DIMS];
      uint8_t m_ndims = 0;
      uint8_t m_nbits = 0;
      uint8_t m_scaling = 1;
      
      AbstractStencil() = default; 
      
      inline bool is_local() const
      {
        for(unsigned int k=0;k<MAX_DIMS && k<m_ndims;k++) if( m_size[k]!=1 ) return false;
        for(unsigned int k=0;k<MAX_DIMS && k<m_ndims;k++) if( m_low[k]!=0 ) return false;
        return true;
      }
      
      inline bool read_bit( size_t i ) const
      {
        assert( /* i>=0 && */ i<MAX_BIT_STORAGE );
        int w = i / 64;
        int b = i % 64;
        return ( m_data[w] >> b ) & uint64_t(1);
      }
      inline void write_bit(size_t i , bool b)
      {
        assert( /* i>=0 && */ i<MAX_BIT_STORAGE );
        int w = i / 64;
        int bi = i % 64;
        uint64_t mask = uint64_t(1) << bi;
        if(b) m_data[w] |= mask;
        else  m_data[w] &= ~mask;
      }
      inline void add_bit(size_t i , bool b)
      {
        assert( /* i>=0 && */ i<MAX_BIT_STORAGE );
        if(b)
        {
          int w = i / 64;
          int bi = i % 64;
          m_data[w] |= uint64_t(1) << bi;
        }
      }
      inline void clear_bits()
      {
        size_t n = nb_cells() * 2 * m_nbits;
        n = std::min( (n+63)/64 , MAX_BIT_STORAGE_WORDS );
        for(size_t i=0;i<n;i++) m_data[i] = 0;
      }

      inline void copy_from( const AbstractStencil& st )
      {
        for(unsigned int k=0;k<MAX_DIMS;k++) m_low[k] = st.m_low[k];
        for(unsigned int k=0;k<MAX_DIMS;k++) m_size[k] = st.m_size[k];
        m_ndims = st.m_ndims;
        m_nbits = st.m_nbits;
        m_scaling = st.m_scaling;
        size_t n = nb_cells() * 2 * m_nbits;
        n = std::min( (n+63)/64 , MAX_BIT_STORAGE_WORDS );
        for(size_t i=0;i<n;i++) m_data[i] = st.m_data[i];
      }

      inline AbstractStencil& operator = ( const AbstractStencil& st )
      {
        copy_from(st);
        return *this;
      }
      
      template<size_t N=MAX_DIMS>
      inline auto low_corner() const
      {
        static_assert(N>=0 && N<=MAX_DIMS);
        oarray_t<ssize_t,N> a; size_t k=0;
        for(;k<m_ndims && k<N;k++) a[k] = m_low[k];
        for(;k<N;k++) a[k] = 0;
        return a;
      }
      
      template<size_t N=MAX_DIMS>
      inline auto box_size() const
      {
        static_assert(N>=0 && N<=MAX_DIMS);
        oarray_t<size_t,N> a; size_t k=0;
        for(;k<m_ndims && k<N;k++) a[k] = m_size[k];
        for(;k<N;k++) a[k] = 1;
        return a;
      }
      inline size_t nb_cells() const { return domain_size( box_size() ); }
      
      template<class A, class S>
      inline void write_stencil_element( const A&, S , int bshift )
      {
        constexpr uint64_t ro_m = dac::DataSlicesSubSet< typename A::item_t , typename S::ro_slices_t >::bit_mask_v;
        constexpr uint64_t rw_m = dac::DataSlicesSubSet< typename A::item_t , typename S::rw_slices_t >::bit_mask_v;
        assert( A::ND == m_ndims );
        oarray_t<int,A::ND> c;
        if constexpr ( S::Nd != 0 )
        {
          static_assert( S::Nd == A::ND , "dimensionality mismatch" );
          c = array_sub( S::relpos_v , m_low );
        }
        if constexpr ( S::Nd == 0 )
        {
          if constexpr (c.size()>0) for(size_t i=0;i<c.size();i++) c[i] = - m_low[i];
        }
        add_ro_mask( ro_m << bshift , c );
        add_rw_mask( rw_m << bshift , c );
      }

      template<class A, class C, size_t S, class... E>
      inline int write_stencil_elements( const A& accessor , Stencil<C,stencil_elements_t<E...> , S > /*stencil_t<C,E...>*/ , int bshift )
      {
        write_stencil_element( accessor , C{} , bshift);
        ( ... , ( write_stencil_element( accessor , E{} , bshift ) ) );
        return bshift + A::nb_slices;
      }

      template<class... T>
      inline AbstractStencil( const FlatTuple<T...>& accessors )
      {
        build_from_accessors( accessors, std::make_index_sequence<sizeof...(T)>{} );
      }


      template<class iseq> struct InegerSequenceHead;
      template<class T, T I0 , T ... Is> struct InegerSequenceHead< std::integer_sequence<T,I0,Is...> > { static inline constexpr T value = I0; };
      //template<class iseq> static inline constexpr typename iseq::value_type integer_sequence_head_v = InegerSequenceHead<iseq>::value;
      template<class iseq> static inline constexpr typename iseq::value_type integer_sequence_head(iseq) { return InegerSequenceHead<iseq>::value; }


      template<class AccTuple, size_t... Is>
      inline void build_from_accessors( const AccTuple& accessors , std::integer_sequence<std::size_t,Is...> iseq)
      {
        if constexpr ( AccTuple::size() >= 1 )
        {
          using A = flat_tuple_element_t<AccTuple, integer_sequence_head(iseq) >;
          constexpr auto nd = A::ND;
          constexpr auto snd = A::access_stencil_t::ndims();
          //constexpr auto a_scaling = A::access_stencil_t::scaling();
          constexpr auto scaling = std::max( { flat_tuple_element_t<AccTuple,Is>::access_stencil_t::scaling() ... } );
          //( ... , ( scaling = std::max(scaling,flat_tuple_element_t<AccTuple,Is>::access_stencil_t::scaling() ) ) );
          
          static_assert( ( ... && ( flat_tuple_element_t<AccTuple,Is>::access_stencil_t::scaling() == scaling || is_local_stencil_v< typename flat_tuple_element_t<AccTuple,Is>::access_stencil_t > ) ) , "inconsistent stencil scaling across accessors" );
          static_assert( snd<=nd && ( snd==0 || snd==nd ) , "stencil dimensionality inconsistent with Accessor's" );
          static_assert( nd <= MAX_DIMS , "dimensionality overload" );
          
          auto ns = 0; // A::nb_slices;
          auto sloc = A::access_stencil_t::low_corner();
          auto shic = A::access_stencil_t::high_corner();

          ( ... , ( ns += flat_tuple_element_t<AccTuple,Is>::nb_slices ) );
          ( ... , ( sloc = array_min(sloc,flat_tuple_element_t<AccTuple,Is>::access_stencil_t::low_corner()) ) );
          ( ... , ( shic = array_max(shic,flat_tuple_element_t<AccTuple,Is>::access_stencil_t::high_corner()) ) );

          static_assert( snd == shic.size() && snd == sloc.size() );
          
          if constexpr (shic.size()>0) for(size_t k=0;k<shic.size();k++) shic[k] = shic[k] + 1 - sloc[k];

          m_ndims = nd;
          m_nbits = ns;
          m_scaling = scaling;
          if constexpr (snd>0 || nd>0)
          {
            size_t k=0;
            if constexpr (snd>0) for(;k<snd;k++)
            {
              m_low[k] = sloc[k];
              m_size[k] = shic[k];
            }
            if constexpr (nd>0) for(;k<nd;k++)
            {
              m_low[k] = 0;
              m_size[k] = 1;
            }
          }
          clear_bits();
          
          ns = 0;
          ( ... , ( ns = write_stencil_elements( accessors.get(tuple_index<Is>) , typename flat_tuple_element_t<AccTuple,Is>::access_stencil_t {} , ns ) ) );
        }
      }
      
      inline uint64_t read_mask(size_t index , size_t a) const
      {
        uint64_t m = 0;
        for(unsigned int i=0;i<m_nbits;i++)
        {
          size_t j = index*m_nbits*2 + a + i;
          uint64_t x = read_bit(j);
          m = (m<<1) | x;
        }
        return m;
      }
      template<size_t N>
      inline uint64_t read_mask( const oarray_t<int,N>& c , size_t a) const
      {
        oarray_t<int,N> dom; for(size_t k=0;k<N;k++) dom[k]=m_size[k];
        return read_mask( coord_to_index(c,dom) , a );
      }      
      template<size_t N> inline uint64_t ro_mask( const oarray_t<int,N>& c ) const { return read_mask(c,0);  }
      template<size_t N> inline uint64_t rw_mask( const oarray_t<int,N>& c ) const { return read_mask(c,m_nbits);  }
      inline uint64_t ro_mask( size_t i ) const { return read_mask(i,0);  }
      inline uint64_t rw_mask( size_t i ) const { return read_mask(i,m_nbits);  }

      inline void add_mask( uint64_t m , size_t index , size_t a)
      {
        //std::cout<<"add mask : m="<<m<<", idx="<<index<<", a="<<a<<std::endl;
        for(unsigned int i=0;i<m_nbits;i++)
        {
          size_t j = index*m_nbits*2 + a + (m_nbits-1-i);
          bool bit = ( m & 1 );
          write_bit( j , read_bit(j) || bit );
          m = m >> 1;
        }
        assert( m == 0 );
      }
      template<size_t N>
      inline void add_mask( uint64_t m , const oarray_t<int,N>& c , size_t a)
      {
        assert( N == m_ndims );
        add_mask( m , coord_to_index( c , m_size ) , a );
      }
      inline void add_ro_mask( uint64_t m , size_t i ) { add_mask(m,i,0);  }
      inline void add_rw_mask( uint64_t m , size_t i ) { add_mask(m,i,m_nbits);  }
      template<size_t N> inline void add_ro_mask( uint64_t m , const oarray_t<int,N>& c ) { add_mask(m,c,0);  }
      template<size_t N> inline void add_rw_mask( uint64_t m , const oarray_t<int,N>& c ) { add_mask(m,c,m_nbits);  }
    };

    // utility methods
    template<size_t Nd> extern std::unordered_set< oarray_t<int,Nd> > stencil_dep_graph( const dac::AbstractStencil & stencil , size_t grainsize = 1 );
    template<size_t Nd> extern std::unordered_set< oarray_t<int,Nd> > stencil_co_dep_graph( const AbstractStencil & stencil, const AbstractStencil & stencil2 , size_t grainsize = 1 );

    extern std::ostream& stencil_dot( std::ostream& out , const AbstractStencil & stencil , const std::function<std::string(uint64_t)> & mask_to_text );
    template<size_t Nd> extern std::ostream& stencil_dep_dot( std::ostream& out , const std::unordered_set< oarray_t<int,Nd> > & dep_rpos );
  }
  // ========================================================
}

namespace std
{
  template<class C, class... E> struct hash< onika::dac::Stencil<C,onika::dac::stencil_elements_t<E...> > > {  inline size_t operator () (const onika::dac::Stencil<C,onika::dac::stencil_elements_t<E...> > & ) const { return 0; } };
}


