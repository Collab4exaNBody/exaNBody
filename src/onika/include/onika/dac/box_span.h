#pragma once

#include <cstdint>
#include <onika/oarray.h>
#include <vector>
#include <onika/oarray_stream.h>
#include <cassert>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>

// =========== data access control constants ===========
namespace onika
{
  namespace dac
  {

    static inline constexpr size_t max_span_border_size = 1ull << 31;
    static inline constexpr size_t span_no_border = max_span_border_size;
  
    template<size_t Nd, size_t _grainsize=1> struct box_span_t
    {
      static inline constexpr size_t ndims = Nd;
      static inline constexpr size_t grainsize = _grainsize;
      using coord_t = oarray_t<size_t,ndims>;
      coord_t lower_bound;
      coord_t box_size;
      size_t border = span_no_border;
      bool _initialized = post_constructor();
      
      ONIKA_HOST_DEVICE_FUNC inline bool post_constructor()
      {
        if constexpr (Nd>0) for(size_t i=0;i<Nd;i++)
        {
          if( box_size[i]==0 ) { box_size[i]=lower_bound[i]; lower_bound[i]=0; }
        }
        return true;
      }
      
      ONIKA_HOST_DEVICE_FUNC inline bool inside(const coord_t& c) const
      {
        assert( c.array_size == ndims );
        ssize_t bd = max_span_border_size;
        for(size_t i=0;i<c.array_size;i++)
        {
          ssize_t x = c[i] - lower_bound[i];
          ssize_t hi = box_size[i] - 1 - x;
          bd = onika::cuda::min( bd , onika::cuda::min( x , hi ) );
        }
        return bd >= 0 && bd < ssize_t(border);
      }
      
      template<class OtherSpanT>
      ONIKA_HOST_DEVICE_FUNC inline void copy_from (const OtherSpanT& sp)
      {
        assert( sp.grainsize == grainsize );
        assert( sp.ndims == ndims );
        if constexpr (ndims>0) for(size_t i=0;i<ndims;i++)
        {
          lower_bound[i] = sp.lower_bound[i];
          box_size[i] = sp.box_size[i];
        }
        border = sp.border;
      }
    };

    template<class T> struct is_span_t : public std::false_type {};
    template<size_t N> struct is_span_t< box_span_t<N> > : public std::true_type {};
    template<class T> static inline constexpr bool is_span_v = is_span_t<T>::value;

    struct abstract_box_span_t
    {
      static inline constexpr size_t MAX_DIMS = 4;
      size_t lower_bound[MAX_DIMS];
      size_t box_size[MAX_DIMS];
      size_t coarse_domain[MAX_DIMS];
      size_t border = span_no_border;
      size_t grainsize = 1;
      unsigned int ndims = 0;
      
      ONIKA_HOST_DEVICE_FUNC inline bool sanity_check() const { return ndims<=MAX_DIMS; }

      abstract_box_span_t() = default;

      template<size_t Nd, size_t _G>
      ONIKA_HOST_DEVICE_FUNC inline abstract_box_span_t( const box_span_t<Nd,_G>& span )
      {
        static_assert( Nd <= MAX_DIMS , "too many span dimensions" );
        assert( Nd == span.ndims );
        ndims = span.ndims;
        border = span.border;
        if( border == 0 ) border = span_no_border;
        grainsize = span.grainsize;
        if( grainsize == 0 ) grainsize = 1;
        assert( border==span_no_border || grainsize==1 );
        size_t i=0;
        if constexpr (Nd>0) for(;i<Nd;i++)
        {
          lower_bound[i] = span.lower_bound[i];
          box_size[i] = span.box_size[i];
          coarse_domain[i] = (box_size[i]+grainsize-1)/grainsize;
        }
        for(;i<MAX_DIMS;i++)
        {
          lower_bound[i] = 0;
          box_size[i] = 1;
          coarse_domain[i] = 1;
        }
        if( hole_cells() == 0 )
        {
          border = span_no_border;
        }
        assert( sanity_check() );
      }

      ONIKA_HOST_DEVICE_FUNC inline bool has_border() const { return border != span_no_border; }

      template<class StreamT>
      inline StreamT& to_stream(StreamT& out) const
      {
        out<<"nd="<<ndims<<",lo="<<format_array(lower_bound,ndims)<<",sz="<<format_array(box_size,ndims)<<",bd=";
        if(border<max_span_border_size) out<<border;
        else out<<"inf";
        out<<" cd="<<format_array(coarse_domain,ndims);
        return out;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,Nd> box_size_nd() const
      {
        assert( Nd == ndims );
        oarray_t<size_t,Nd> a;
        for(size_t k=0;k<Nd;k++) a[k]=box_size[k];
        return a;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,Nd> lower_bound_nd() const
      {
        assert( Nd == ndims );
        oarray_t<size_t,Nd> a;
        for(size_t k=0;k<Nd;k++) a[k]=lower_bound[k];
        return a;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,Nd> coarse_domain_nd() const
      {
        assert( Nd == ndims );
        oarray_t<size_t,Nd> a;
        for(size_t k=0;k<Nd;k++) a[k]=coarse_domain[k];
        return a;
      }

      ONIKA_HOST_DEVICE_FUNC inline size_t box_cells() const
      {
        assert( ndims > 0 );
        size_t n_cells = box_size[0];
        for(size_t i=1;i<MAX_DIMS && i<ndims;i++)
        {
          n_cells *= box_size[i];
        }
        return n_cells;
      }

      ONIKA_HOST_DEVICE_FUNC inline size_t hole_cells() const
      {
        // assert( ndims >= 0 );
        if( ndims==0 ) return 0;
        ssize_t h = onika::cuda::max( ssize_t(0) , ssize_t(box_size[0]) - ssize_t(2*border) );
        for(size_t i=1;i<MAX_DIMS && i<ndims;i++)
        {
          h *= onika::cuda::max( ssize_t(0) , ssize_t(box_size[i]) - ssize_t(2*border) );
        }
        return h;
      }

      ONIKA_HOST_DEVICE_FUNC inline size_t nb_cells() const { return box_cells() - hole_cells(); }
      ONIKA_HOST_DEVICE_FUNC inline size_t nb_coarse_cells() const
      {
        if( grainsize==1 ) { return nb_cells(); }
        assert( border == span_no_border );
        size_t n = coarse_domain[0];
        for(size_t i=1;i<MAX_DIMS && i<ndims;i++) n *= coarse_domain[i];
        return n;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,Nd> coarse_index_to_coord_base(size_t i) const
      {
        auto c = ::onika::index_to_coord(i,coarse_domain_nd<Nd>());
        for(size_t k=0;k<Nd;k++) c[k] = c[k]*grainsize + lower_bound[k];
        return c;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline size_t coord_to_index(const oarray_t<size_t,Nd>& _c) const
      {
        assert( Nd == ndims );
        assert( inside(_c) );
        auto rbox_low = lower_bound_nd<Nd>();
        auto rbsize = box_size_nd<Nd>();
        auto c = array_sub( _c , rbox_low );
        if( ! has_border() )
        {
          return ::onika::coord_to_index( c , rbsize );
        }
        size_t x = 0;
        for(size_t d=0;d<Nd;d++) // removing th Nd-d-1 dimension at each iteration
        {
          auto slice = rbsize;
          slice[Nd-d-1] = border;
          size_t slice_cells = domain_size( slice );
          if( c[Nd-d-1] < border )
          {
            return x + ::onika::coord_to_index( c , slice );
          }
          //std::cout<<"x="<<x<<" d="<<d<<" slice_cells="<<slice_cells<<" n="<<n<<std::endl;
          x += slice_cells;
          c[Nd-d-1] -= border;
          rbsize[Nd-d-1] -= 2*border;
          if( ssize_t(c[Nd-d-1]) >= ssize_t(rbsize[Nd-d-1]) )
          {
            c[Nd-d-1] -= rbsize[Nd-d-1];
            return x + ::onika::coord_to_index( c , slice );
          }
          x += slice_cells;
        }
        ONIKA_CU_ABORT();
        return 0;
      }

      template<size_t Nd>
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,Nd> index_to_coord(size_t x) const
      {
        assert( Nd == ndims );
        auto rbsize = box_size_nd<Nd>();
        auto rbox_low = lower_bound_nd<Nd>();
        if( ! has_border() )
        {
          return array_add( rbox_low , ::onika::index_to_coord( x , rbsize ) );
        }
        for(size_t d=0;d<Nd;d++) // removing th Nd-d-1 dimension at each iteration
        {
          auto slice = rbsize;
          slice[Nd-d-1] = border;
          size_t slice_cells = domain_size( slice );
          if( x < slice_cells )
          {
            return array_add( rbox_low , ::onika::index_to_coord(x,slice) );
          }
          //std::cout<<"x="<<x<<" d="<<d<<" slice_cells="<<slice_cells<<" n="<<n<<std::endl;
          x -= slice_cells;
          if( x < slice_cells )
          {
            auto c = array_add( rbox_low , ::onika::index_to_coord(x,slice) );
            c[Nd-d-1] += rbsize[Nd-d-1] - border;
            return c;
          }
          x -= slice_cells;
          rbox_low[Nd-d-1] += border;
          rbsize[Nd-d-1] -= 2*border;
        }
        // ERROR
        //std::abort();
        for(auto& c:rbox_low) c = onika::cuda::numeric_limits<size_t>::max;
        return rbox_low;
      }

      template<class T, size_t N>
      ONIKA_HOST_DEVICE_FUNC inline bool inside(const oarray_t<T,N>& c) const
      {
        assert( c.size() == ndims );
        ssize_t bd = max_span_border_size;
        for(size_t i=0;i<c.size();i++)
        {
          ssize_t x = c[i] - lower_bound[i];
          ssize_t hi = box_size[i] - 1 - x;
          bd = onika::cuda::min( bd , onika::cuda::min( x , hi ) );
        }
        return bd >= 0 && bd < ssize_t(border);
      }

      ONIKA_HOST_DEVICE_FUNC inline bool operator == ( const abstract_box_span_t& sp ) const
      {
        if( sp.ndims != ndims ) return false;
        if( sp.border != border ) return false;
        if( sp.grainsize != grainsize ) return false;
        for(unsigned int i=0;i<ndims;i++) if( sp.lower_bound[i] != lower_bound[i] || sp.box_size[i] != box_size[i] ) return false;
        return true;
      }
    };

    ONIKA_HOST_DEVICE_FUNC inline abstract_box_span_t bounding_span(const abstract_box_span_t& sp1 , const abstract_box_span_t& sp2)
    {
      assert( sp1.ndims == sp2.ndims );
      assert( sp1.border == sp2.border && sp1.border == onika::cuda::numeric_limits<size_t>::max );
      abstract_box_span_t r;
      r.ndims = sp1.ndims;
      for(unsigned int k=0;k<r.ndims;k++)
      {
        r.lower_bound[k] = onika::cuda::min( sp1.lower_bound[k] , sp2.lower_bound[k] );
        r.box_size[k] = onika::cuda::max( sp1.box_size[k] + sp1.lower_bound[k] , sp2.box_size[k] + sp2.lower_bound[k] ) - r.lower_bound[k];
      }
      assert( r.border == onika::cuda::numeric_limits<size_t>::max );
      return r;
    }
  // ========================================================
  }  

}

#include <functional>

namespace std
{
  template<size_t Nd, size_t G> struct hash< onika::dac::box_span_t<Nd,G> >
  {
    inline size_t operator () ( const onika::dac::box_span_t<Nd,G> & sp ) const
    {
      return std::hash<std::string_view>{}( std::string_view( (const char*) &sp , sizeof(onika::dac::box_span_t<Nd,G>) ) );
    }
  };

}



